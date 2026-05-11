/**********************************************************************
  Unfolding_Bands_fast.c

     Unfolding_Bands_fast.c is a subroutine to calculate unfolded weight
     at given k-points for the file output.

  Log of Band_Unfolding.c:

      6/Jan/2016  Released by Chi-Cheng Lee
      23/Dec/2023  Modified by M. FUKUDA
          - Added update_mapN2n
          - Reconstruct the algorithm for calculating weight.
          - MPI parallelization.


***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static int Norb;
static int natom;
static int* Norbperatom;
static int*** tabr4RN;
static int** mapN2rn;
static int exitcode;
static int totnkpts;
static int* np;
static int** mapn2N;
static int* num_mapn2N;

static double tv_c[4][4];
static double rtv_c[4][4];
static int CpyCell_c, TCpyCell_c;
static double** atv_c;
static int** atv_c_ijk;
static int*** ratv_c;
static double Cell_Volume_c;

static int nkpath;
static double*** kpath;
static char*** kname;

static double *T_KGrids1;
static double *T_KGrids2;
static double *T_KGrids3;
static double *T_kGrids1;
static double *T_kGrids2;
static double *T_kGrids3;
static double *T_kdis;
static double **T_kpath_kdis;

static void set_nkpath();
static void set_kpath();
static void free_kpath();
static void determine_kpts();
static void set_KGrids();
static void free_KGrids();
static double volume(const double* a,const double* b,const double* c);
static double dot(const double* v1,const double* v2);
static double distwovec(const double* a,const double* b);
static void getnorbperatom();
static void abc_by_ABC(double** S);

static void Merge_unfolding_output(const char* file_EV, const int digit, const int numprocs);
static void update_mapN2n(int* mapN2n);
static void set_atv_for_conceptual_cell(double** lattice, double* origin, int* mapN2n);
static int  set_tabr4RN(int*** tabr4RN, double* auto_origin, double* mergin);
static void free_atv_for_conceptual_cell();
static void generate_gnuplot_example(int SpinP_switch);

static void Unfolding_Bands_Col(
                                int SpinP_switch, 
                                double *****nh,
                                double ****CntOLP);

static void Unfolding_Bands_NonCol(
                                   int SpinP_switch, 
                                   double *****nh,
                                   double *****ImNL,
                                   double ****CntOLP);


void Unfolding_Bands( 
                      int SpinP_switch, 
                      double *****nh,
                      double *****ImNL,
                      double ****CntOLP)
{
  if (SpinP_switch==0 || SpinP_switch==1){
    Unfolding_Bands_Col( SpinP_switch, nh, CntOLP);
  }
  else if (SpinP_switch==3){
    Unfolding_Bands_NonCol( SpinP_switch, nh, ImNL, CntOLP);
  }
}



static void Unfolding_Bands_Col(
                                int SpinP_switch, 
                                double *****nh,
                                double ****CntOLP)
{

  double coe;
  double* a;
  double* b;
  double* c;
  double* K;
  double r[3];
  double r0[3];
  int rn, rn0;
  double kj_e;
  dcomplex** weight;
  dcomplex*** tmp_weight;
  dcomplex** kj_v;

  int i,j,k,l,n;
  int *MP,*order_GA,*My_NZeros,*SP_NZeros,*SP_Atoms;
  int i1,j1,spin,size_H1;
  int l1,l2,l3,kloop;
  int h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum;
  int ii,Rn,AN;
  int mul,m,wan1,TNO1,Gc_AN;
  int Gh_AN, wan2, TNO2;
  int LB_AN,GB_AN,Bnum;

  double tmp,av_num;
  double k1,k2,k3,sum,sumi;
  double EV_cut0;
  double **ko,*M1,**EIGEN;
  double *koS;
  double *S1,**H1;
  dcomplex ***H,**S,***C;
  dcomplex Ctmp1,Ctmp2;
  double kRn,si,co;
  double TStime,TEtime,SiloopTime,EiloopTime;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV;
  FILE *fp_EV1;
  FILE *fp_EV2;
  FILE *fp_EV3;
  //char buf[fp_bsize];          /* setvbuf */
  //char buf1[fp_bsize];          /* setvbuf */
  //char buf2[fp_bsize];          /* setvbuf */
  //char buf3[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID,digit,fd;
  int Nthrds, index_thread;

  int boundary_id;
  int *is1,*ie1;
  double Stime,Etime;
  int kloopi,kloopj;
  int* array_kloopi;
  int* array_kloopj;
  int iloop,total_iloop;
  int iloop_start, iloop_end;
  int* array_spin;
  int* array_i;
  int* count_k4S;


  int MA,MB,MO,NO, mb;
  dcomplex dtmp;
  dcomplex phase1,phase2;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  dtime(&Stime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) {
    printf("\n*******************************************************\n");
    printf("                 Unfolding of Bands \n");
    printf("*******************************************************\n\n");fflush(stdout);
  } 
  dtime(&TStime);

  update_mapN2n(unfold_mapN2n);

  /****************************************************
                  allocation of arrays
  ****************************************************/
  
  getnorbperatom();

  set_nkpath();
  set_kpath();

  exitcode=0;

  set_atv_for_conceptual_cell(unfold_abc,unfold_origin,unfold_mapN2n);

  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    return;
  }

  coe=Cell_Volume/Cell_Volume_c;
  determine_kpts();

  /*****************************************************
        allocation of arrays for eigenvalue problem 
  *****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));
  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }

  ko = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    ko[i] = (double*)malloc(sizeof(double)*(n+1));
  }

  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
  for (j=0; j<List_YOUSO[23]; j++){
    EIGEN[j] = (double*)malloc(sizeof(double)*(n+1));
  }

  H = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    H[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      H[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
  for (i=0; i<n+1; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
  }

  M1 = (double*)malloc(sizeof(double)*(n+1));

  C = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    C[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      C[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of S1 and H1 */
  S1 = (double*)malloc(sizeof(double)*size_H1);
  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H1[spin] = (double*)malloc(sizeof(double)*size_H1);
  }

  /* Get S1 */
  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, nh[1], H1[1], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  count_k4S = (int*)malloc(sizeof(int)*atomnum);

  k=0;
  for (MA=0; MA<atomnum; MA++) {
    count_k4S[MA] = k;
    GA_AN = order_GA[MA+1];
    wan1 = WhatSpecies[GA_AN];
    TNO1 = Spe_Total_CNO[wan1];
    for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++) {
      Gh_AN = natn[GA_AN][h_AN];
      wan2 = WhatSpecies[Gh_AN];
      TNO2 = Spe_Total_CNO[wan2];

      k = k + TNO1*TNO2;
    }
  }

  K=(double*)malloc(sizeof(double)*3);

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  set_KGrids();

#pragma omp parallel
  {
    Nthrds = omp_get_num_threads();
  }

  tmp_weight=(dcomplex***)malloc(sizeof(dcomplex**)*Nthrds);
  for (i=0; i<Nthrds; i++){
    tmp_weight[i]=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
    for (j=0; j<atomnum; j++){
      tmp_weight[i][j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
    }
  }

  weight=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (i=0; i<atomnum; i++) weight[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);

  kj_v=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (j=0; j<atomnum; j++) kj_v[j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);

  Name_Angular[0][0] = "s          ";
  Name_Angular[1][0] = "px         ";
  Name_Angular[1][1] = "py         ";
  Name_Angular[1][2] = "pz         ";
  Name_Angular[2][0] = "d3z^2-r^2  ";
  Name_Angular[2][1] = "dx^2-y^2   ";
  Name_Angular[2][2] = "dxy        ";
  Name_Angular[2][3] = "dxz        ";
  Name_Angular[2][4] = "dyz        ";
  Name_Angular[3][0] = "f5z^2-3r^2 ";
  Name_Angular[3][1] = "f5xz^2-xr^2";
  Name_Angular[3][2] = "f5yz^2-yr^2";
  Name_Angular[3][3] = "fzx^2-zy^2 ";
  Name_Angular[3][4] = "fxyz       ";
  Name_Angular[3][5] = "fx^3-3*xy^2";
  Name_Angular[3][6] = "f3yx^2-y^3 ";
  Name_Angular[4][0] = "g1         ";
  Name_Angular[4][1] = "g2         ";
  Name_Angular[4][2] = "g3         ";
  Name_Angular[4][3] = "g4         ";
  Name_Angular[4][4] = "g5         ";
  Name_Angular[4][5] = "g6         ";
  Name_Angular[4][6] = "g7         ";
  Name_Angular[4][7] = "g8         ";
  Name_Angular[4][8] = "g9         ";

  Name_Multiple[0] = "0";
  Name_Multiple[1] = "1";
  Name_Multiple[2] = "2";
  Name_Multiple[3] = "3";
  Name_Multiple[4] = "4";
  Name_Multiple[5] = "5";

  if (myid==Host_ID){
    strcpy(file_EV,".EV");
    fnjoint(filepath,filename,file_EV);
    if ((fp_EV = fopen(file_EV,"a")) != NULL){
      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"          Unfolding calculation for band structure         \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"                                                                          \n");
      fprintf(fp_EV," Origin of the Reference cell is set to (%f %f %f) (Bohr).\n\n",
              unfold_origin[0],unfold_origin[1],unfold_origin[2]);
      fprintf(fp_EV," Unfolded weights at specified k points are stored in System.Name.unfold_totup(dn).\n");
      fprintf(fp_EV," Individual orbital weights are stored in System.Name.unfold_orbup(dn).\n");
      fprintf(fp_EV," The format is: k_dis(Bohr^{-1})  energy(eV)  weight.\n\n");
      fprintf(fp_EV," The sequence for the orbital weights in System.Name.unfold_orbup(dn) is given below.\n\n");

      i1 = 1;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
        for (l=0; l<=Supported_MaxL; l++){
          for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
            for (m=0; m<(2*l+1); m++){
              fprintf(fp_EV,"  %4d ",i1);
              if (l==0 && mul==0 && m==0)
                fprintf(fp_EV,"%4d %3s %s %s",
                        Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
              else
                fprintf(fp_EV,"         %s %s",
                        Name_Multiple[mul],Name_Angular[l][m]);
              fprintf(fp_EV,"\n");
              i1++;
            }
          }
        }
      }

      fprintf(fp_EV,"\n"); 
      fprintf(fp_EV,"\n  The total number of calculated k points is %i.\n",totnkpts);
      fprintf(fp_EV,"  The number of calculated k points on each path is \n");
 
      fprintf(fp_EV,"  For each path: ("); 
      for (i=1; i<=nkpath; i++){
        fprintf(fp_EV," %i",np[i]); 
      }
      fprintf(fp_EV," )\n\n");

      fprintf(fp_EV,"                 ka         kb         kc\n");
      fflush(stdout);

      for (kloop=1; kloop<=totnkpts; kloop++){
        fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);
      }

      fprintf(fp_EV,"\n");
      fclose(fp_EV);

    }
    else{
      printf("Failure of saving the EV file.\n");
      fclose(fp_EV);
    }
  }

  if (SpinP_switch==0) {

    //strcpy(file_EV,".unfold_totup");
    sprintf(file_EV,".unfold_totup_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);

    fp_EV = fopen(file_EV,"w");
    if (fp_EV == NULL) {
      printf("Failure of saving the System.Name.unfold_totup file.\n");
      fclose(fp_EV);
    }

    //strcpy(file_EV,".unfold_orbup");
    sprintf(file_EV,".unfold_orbup_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);
    fp_EV1 = fopen(file_EV,"w");

    if (fp_EV1 == NULL) {
      printf("Failure of saving the System.Name.unfold_orbup file.\n");
      fclose(fp_EV1);
    }

  } 
  else if (SpinP_switch==1) {
    //strcpy(file_EV,".unfold_totup");
    sprintf(file_EV,".unfold_totup_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);
    fp_EV = fopen(file_EV,"w");
    if (fp_EV == NULL) {
      printf("Failure of saving the System.Name.unfold_totup file.\n");
      fclose(fp_EV);
    }
    //strcpy(file_EV,".unfold_orbup");
    sprintf(file_EV,".unfold_orbup_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);
    fp_EV1 = fopen(file_EV,"w");
    if (fp_EV1 == NULL) {
      printf("Failure of saving the System.Name.unfold_orbup file.\n");
      fclose(fp_EV1);
    }
    //strcpy(file_EV,".unfold_totdn");
    sprintf(file_EV,".unfold_totdn_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);
    fp_EV2 = fopen(file_EV,"w");
    if (fp_EV2 == NULL) {
      printf("Failure of saving the System.Name.unfold_totdn file.\n");
      fclose(fp_EV2);
    }
    //strcpy(file_EV,".unfold_orbdn");
    sprintf(file_EV,".unfold_orbdn_%0*i",digit,myid);
    fnjoint(filepath,filename,file_EV);
    fp_EV3 = fopen(file_EV,"w");
    if (fp_EV3 == NULL) {
      printf("Failure of saving the System.Name.unfold_orbdn file.\n");
      fclose(fp_EV3);
    }
  }

  /* for gnuplot example */
  if (myid==Host_ID){
    generate_gnuplot_example(SpinP_switch);
  }
  /* end gnuplot example */

  /* for standard output */

  if (myid==Host_ID && 0<level_stdout) {
    printf(" The number of selected k points is %i.\n",totnkpts);

    printf(" For each path: (");
    for (i=1; i<=nkpath; i++){
      printf(" %i",np[i]); 
    }
    printf(" )\n\n");
    printf("                 ka         kb         kc\n");
  }

  MPI_Barrier(mpi_comm_level1);

  /*********************************************
                      kloopi 
  *********************************************/

  array_kloopi=(int*)malloc(sizeof(int)*(totnkpts+1));
  array_kloopj=(int*)malloc(sizeof(int)*(totnkpts+1));

  kloopi = 0;
  kloopj = 0;
  for (kloop=1; kloop<=totnkpts; kloop++) {
      array_kloopi[kloop] = kloopi;
      array_kloopj[kloop] = kloopj;
      //printf(kloop, array_kloopi[kloop], array_kloopj[kloop]);

      kloopj++;
      if(kloopj == np[kloopi]){
        kloopi++;
        kloopj = 0;
      }
  }

  /* MPI for kloop */
  //int kloop_start, kloop_end;
  //if (numprocs > totnkpts) {
  //  if (myid < numprocs){
  //    kloop_start = myid;
  //    kloop_end = kloop_start;
  //  }
  //  else {
  //      kloop_start = 0;
  //      kloop_end = -1;
  //  }
  //}
  //else {
  //  boundary_id_k = totnkpts%numprocs;
  //  if ( myid < boundary_id_k ){
  //    kloop_start = myid*(totnkpts/numprocs + 1);
  //    kloop_end = kloop_start + totnkpts/numprocs;
  //  }
  //  else {
  //    kloop_start = boundary_id_k*(totnkpts/numprocs + 1) + (myid - boundary_id_k)*(totnkpts/numprocs);
  //    kloop_end = kloop_start + totnkpts/numprocs - 1;
  //  }
  //}

  //printf("myid, kloop_start, kloop_end = %d %d %d\n", myid, kloop_start, kloop_end);fflush(stdout);

  //for (kloop=kloop_start; kloop<=kloop_end; kloop++)
  for (kloop=1; kloop<=totnkpts; kloop++) {
      kloopi = array_kloopi[kloop];
      kloopj = array_kloopj[kloop];
    
      /* for standard output */
     
      if (myid==Host_ID && 0<level_stdout) {

        if (kloop==totnkpts)
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);
        else 
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);

        fflush(stdout);
      }

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S */

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          S[i1][j1] = Complex(0.0,0.0);
        } 
      } 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
        GA_AN = order_GA[AN];
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];

        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
          GB_AN = natn[GA_AN][LB_AN];
          Rn = ncn[GA_AN][LB_AN];
          wanB = WhatSpecies[GB_AN];
          tnoB = Spe_Total_CNO[wanB];
          Bnum = MP[GB_AN];

          l1 = atv_ijk[Rn][1];
          l2 = atv_ijk[Rn][2];
          l3 = atv_ijk[Rn][3];
          kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

          si = sin(2.0*PI*kRn);
          co = cos(2.0*PI*kRn);

          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB; j++){

              S[Anum+i][Bnum+j].r += S1[k]*co;
              S[Anum+i][Bnum+j].i += S1[k]*si;

              k++;
            }
          }
        }
      }

      /* diagonalization of S */
      Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

      if (3<=level_stdout){
        printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",kloop,k1,k2,k3);
        for (i1=1; i1<=n; i1++){
          printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,koS[i1]);
        }
      }

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
        if (koS[l]<0.0) koS[l] = 1.0e-14;
      }

      /* calculate S*1/sqrt(koS) */

      for (l=1; l<=n; l++) M1[l] = 1.0/sqrt(koS[l]);

      /* S * M1  */

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          S[i1][j1].r = S[i1][j1].r*M1[j1];
          S[i1][j1].i = S[i1][j1].i*M1[j1];
        } 
      } 

      /* loop for spin */

      for (spin=0; spin<=SpinP_switch; spin++){

        /* make H */

        for (i1=1; i1<=n; i1++){
          for (j1=1; j1<=n; j1++){
            H[spin][i1][j1] = Complex(0.0,0.0);
          } 
        } 

        k = 0;
        for (AN=1; AN<=atomnum; AN++){
          GA_AN = order_GA[AN];
          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            GB_AN = natn[GA_AN][LB_AN];
            Rn = ncn[GA_AN][LB_AN];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            Bnum = MP[GB_AN];

            l1 = atv_ijk[Rn][1];
            l2 = atv_ijk[Rn][2];
            l3 = atv_ijk[Rn][3];
            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

            for (i=0; i<tnoA; i++){

              for (j=0; j<tnoB; j++){

                H[spin][Anum+i][Bnum+j].r += H1[spin][k]*co;
                H[spin][Anum+i][Bnum+j].i += H1[spin][k]*si;

                k++;

              }
            }
          }
        }

        /* first transpose of S */

        for (i1=1; i1<=n; i1++){
          for (j1=i1+1; j1<=n; j1++){
            Ctmp1 = S[i1][j1];
            Ctmp2 = S[j1][i1];
            S[i1][j1] = Ctmp2;
            S[j1][i1] = Ctmp1;
          }
        }

        /****************************************************
                      M1 * U^t * H * U * M1
        ****************************************************/

        /* H * U * M1 */

#pragma omp parallel for shared(spin,n,myid,is1,ie1,S,H,C) private(i1,j1,l) 

        for (j1=is1[myid]; j1<=ie1[myid]; j1++){

          for (i1=1; i1<=(n-1); i1+=2){

            double sum0  = 0.0, sum1  = 0.0;
            double sumi0 = 0.0, sumi1 = 0.0;

            for (l=1; l<=n; l++){
              sum0  += H[spin][i1+0][l].r*S[j1][l].r - H[spin][i1+0][l].i*S[j1][l].i;
              sum1  += H[spin][i1+1][l].r*S[j1][l].r - H[spin][i1+1][l].i*S[j1][l].i;

              sumi0 += H[spin][i1+0][l].r*S[j1][l].i + H[spin][i1+0][l].i*S[j1][l].r;
              sumi1 += H[spin][i1+1][l].r*S[j1][l].i + H[spin][i1+1][l].i*S[j1][l].r;
            }

            C[spin][j1][i1+0].r = sum0;
            C[spin][j1][i1+1].r = sum1;

            C[spin][j1][i1+0].i = sumi0;
            C[spin][j1][i1+1].i = sumi1;
          }

          for (; i1<=n; i1++){

            double sum  = 0.0;
            double sumi = 0.0;

            for (l=1; l<=n; l++){
              sum  += H[spin][i1][l].r*S[j1][l].r - H[spin][i1][l].i*S[j1][l].i;
              sumi += H[spin][i1][l].r*S[j1][l].i + H[spin][i1][l].i*S[j1][l].r;
            }

            C[spin][j1][i1].r = sum;
            C[spin][j1][i1].i = sumi;
          }

        } /* i1 */ 

        /* M1 * U^+ H * U * M1 */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l)  

        for (i1=1; i1<=n; i1++){
          for (j1=is1[myid]; j1<=ie1[myid]; j1++){
  
            double sum  = 0.0;
            double sumi = 0.0;

            for (l=1; l<=n; l++){
              sum  +=  S[i1][l].r*C[spin][j1][l].r + S[i1][l].i*C[spin][j1][l].i;
              sumi +=  S[i1][l].r*C[spin][j1][l].i - S[i1][l].i*C[spin][j1][l].r;
            }

            H[spin][j1][i1].r = sum;
            H[spin][j1][i1].i = sumi;

          }
        } 

        /* broadcast H */

        BroadCast_ComplexMatrix(mpi_comm_level1,H[spin],n,is1,ie1,myid,numprocs,
                                stat_send,request_send,request_recv);

        /* H to C */

        for (i1=1; i1<=n; i1++){
          for (j1=1; j1<=n; j1++){
            C[spin][j1][i1] = H[spin][i1][j1];
          }
        }

        /* penalty for ill-conditioning states */

        EV_cut0 = 1.0e-9;

        for (i1=1; i1<=n; i1++){

          if (koS[i1]<EV_cut0){
            C[spin][i1][i1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
          }
 
          /* cutoff the interaction between the ill-conditioned state */
 
          if (1.0e+3<C[spin][i1][i1].r){
            for (j1=1; j1<=n; j1++){
              C[spin][i1][j1] = Complex(0.0,0.0);
              C[spin][j1][i1] = Complex(0.0,0.0);
            }
            C[spin][i1][i1].r = 1.0e+4;
          }
        }

        /* diagonalization of C */
        Eigen_PHH(mpi_comm_level1,C[spin],ko[spin],n,n,0);

        for (i1=1; i1<=n; i1++){
          EIGEN[spin][i1] = ko[spin][i1];
        }

        /****************************************************
          transformation to the original eigenvectors.
                 NOTE JRCAT-244p and JAIST-2122p 
        ****************************************************/

        /*  The H matrix is distributed by row */

        for (i1=1; i1<=n; i1++){
          for (j1=is1[myid]; j1<=ie1[myid]; j1++){
            H[spin][j1][i1] = C[spin][i1][j1];
          }
        }

        /* second transpose of S */

        for (i1=1; i1<=n; i1++){
          for (j1=i1+1; j1<=n; j1++){
            Ctmp1 = S[i1][j1];
            Ctmp2 = S[j1][i1];
            S[i1][j1] = Ctmp2;
            S[j1][i1] = Ctmp1;
          }
        }

        /* C is distributed by row in each processor */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l,sum,sumi)  

        for (j1=is1[myid]; j1<=ie1[myid]; j1++){
          for (i1=1; i1<=n; i1++){

            sum  = 0.0;
            sumi = 0.0;
            for (l=1; l<=n; l++){
              sum  += S[i1][l].r*H[spin][j1][l].r - S[i1][l].i*H[spin][j1][l].i;
              sumi += S[i1][l].r*H[spin][j1][l].i + S[i1][l].i*H[spin][j1][l].r;
            }

            C[spin][j1][i1].r = sum;
            C[spin][j1][i1].i = sumi;
          }
        }

        /* broadcast C:
           C is distributed by row in each processor
        */

        BroadCast_ComplexMatrix(mpi_comm_level1,C[spin],n,is1,ie1,myid,numprocs,
                                stat_send,request_send,request_recv);

      } /* spin */

      MPI_Barrier(mpi_comm_level1);

      /****************************************************
                          Output
      ****************************************************/


      //if (myid==Host_ID){

        //setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
        //setvbuf(fp_EV1,buf1,_IOFBF,fp_bsize);  /* setvbuf */
        //setvbuf(fp_EV2,buf2,_IOFBF,fp_bsize);  /* setvbuf */
        //setvbuf(fp_EV3,buf3,_IOFBF,fp_bsize);  /* setvbuf */

        
        K[0]=T_KGrids1[kloop]*rtv[1][1]+T_KGrids2[kloop]*rtv[2][1]+T_KGrids3[kloop]*rtv[3][1];
        K[1]=T_KGrids1[kloop]*rtv[1][2]+T_KGrids2[kloop]*rtv[2][2]+T_KGrids3[kloop]*rtv[3][2];
        K[2]=T_KGrids1[kloop]*rtv[1][3]+T_KGrids2[kloop]*rtv[2][3]+T_KGrids3[kloop]*rtv[3][3];

        iloop = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (i=1; i<=n; i++){
            if (((EIGEN[spin][i]-ChemP)<=unfold_ubound)&&((EIGEN[spin][i]-ChemP)>=unfold_lbound)) {
              iloop++;
            }
          }
        }
        total_iloop = iloop;

        array_spin=(int*)malloc(sizeof(int)*total_iloop);
        array_i=(int*)malloc(sizeof(int)*total_iloop);

        iloop = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (i=1; i<=n; i++){
            if (((EIGEN[spin][i]-ChemP)<=unfold_ubound)&&((EIGEN[spin][i]-ChemP)>=unfold_lbound)) {
              array_spin[iloop] = spin;
              array_i[iloop] = i;
              //printf("iloop, spin, i = %d %d %d\n",iloop, spin, i);
              iloop++;
            }
          }
        }

        /* MPI for iloop */
        if (numprocs > total_iloop) {
          if (myid < total_iloop){
            iloop_start = myid;
            iloop_end = iloop_start;
          }
          else {
              iloop_start = 0;
              iloop_end = -1;
          }
        }
        else {
          boundary_id = total_iloop%numprocs;
          if ( myid < boundary_id ){
            iloop_start = myid*(total_iloop/numprocs + 1);
            iloop_end = iloop_start + total_iloop/numprocs;
          }
          else {
            iloop_start = boundary_id*(total_iloop/numprocs + 1) + (myid - boundary_id)*(total_iloop/numprocs);
            iloop_end = iloop_start + total_iloop/numprocs - 1;
          }
        }
        //printf("myid, iloop_start, iloop_end, total_iloop = %d, %d, %d, %d\n",myid, iloop_start, iloop_end, total_iloop);fflush(stdout);

        for (iloop=iloop_start; iloop<=iloop_end; iloop++){
          spin = array_spin[iloop];
          i = array_i[iloop];
          //printf("myid, iloop, spin, i = %d, %d, %d, %d\n",myid, iloop, spin, i);fflush(stdout);
          kj_e=(EIGEN[spin][i]-ChemP);

          i1 = 1; 
          int iorb;

          for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
            iorb=0;

            wan1 = WhatSpecies[Gc_AN];
          
            for (l=0; l<=Supported_MaxL; l++){
              for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
                for (m=0; m<(2*l+1); m++){
                  kj_v[Gc_AN-1][iorb]=Complex(C[spin][i][i1].r,C[spin][i][i1].i);

                  i1++;
                  iorb++;
                }
              }
            }
          }

          //printf("myid, Nthrds =%d\n",myid, Nthrds);
          for (l=0; l<Nthrds; l++) for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) tmp_weight[l][k][j]=Complex(0.,0.);
          for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight[k][j]=Complex(0.,0.);


#pragma omp parallel for \
  shared( count_k4S, atomnum, order_GA, WhatSpecies, Spe_Total_CNO, FNAN,\
         natn, ncn, mapN2rn, num_mapn2N, mapn2N, a, b, c, K,\
         tmp_weight, unfold_mapN2n, kj_v, S1, myid, atv_c)\
  private( index_thread, k, MA, GA_AN, wan1, TNO1, h_AN, Gh_AN, Rn, wan2, TNO2,\
           r, rn, rn0, ii, l, MB, mb, r0, phase1, phase2, MO, NO, dtmp)
          for (MA=0; MA<atomnum; MA++) {
            index_thread = omp_get_thread_num();
            k = count_k4S[MA];
            GA_AN = order_GA[MA+1];
            wan1 = WhatSpecies[GA_AN];
            TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++) {
              Gh_AN = natn[GA_AN][h_AN];
              Rn = ncn[GA_AN][h_AN];
              wan2 = WhatSpecies[Gh_AN];
              TNO2 = Spe_Total_CNO[wan2];

              rn = mapN2rn[Gh_AN-1][Rn];
              r[0] = atv_c[rn][1];
              r[1] = atv_c[rn][2];
              r[2] = atv_c[rn][3];

              phase2=Cexp(Complex(0.,dot(K,r)));

              mb = unfold_mapN2n[Gh_AN-1];

              for (ii=0; ii<num_mapn2N[mb]; ii++){
                MB = mapn2N[mb][ii];
                rn0 = mapN2rn[MB][0];
                r0[0] = atv_c[rn0][1];
                r0[1] = atv_c[rn0][2];
                r0[2] = atv_c[rn0][3];

                phase1=Cmul(phase2,Cexp(Complex(0.,-dot(K,r0))));

                l=0;
                for (MO=0; MO<TNO1; MO++){
                  for (NO=0; NO<TNO2; NO++) {
                    dtmp=Cmul(Conjg(kj_v[GA_AN-1][MO]),kj_v[MB][NO]);
                    dtmp=RCmul(S1[k+l],dtmp);
                    dtmp=Cmul(phase1,dtmp);
                    tmp_weight[index_thread][MB][NO] = Cadd(tmp_weight[index_thread][MB][NO],dtmp);
                    l++;
                  }
                }
              }
              k = k + TNO1*TNO2;
            }
          }
          
          for (l=0; l<Nthrds; l++){
            for (j=0; j<atomnum; j++){
              for (k=0; k<Norbperatom[j]; k++){
                 weight[j][k] = Cadd(weight[j][k],tmp_weight[l][j][k]);
              }
            }
          }

          double sumallorb=0.;

          for (j=0; j<atomnum; j++){
            for (k=0; k<Norbperatom[j]; k++){
               sumallorb += weight[j][k].r;
            }
          }

          if (spin==0) {
            fprintf(fp_EV,"%f %f %10.7f\n",T_kdis[kloop],kj_e*eV2Hartree,fabs(sumallorb)/coe);
          }
          else {
            fprintf(fp_EV2,"%f %f %10.7f\n",T_kdis[kloop],kj_e*eV2Hartree,fabs(sumallorb)/coe);
          }

          /* set negative weight to zero for plotting purpose */
          for (j=0; j<atomnum; j++){
            for (k=0; k<Norbperatom[j]; k++){
               if (weight[j][k].r<0.0) weight[j][k].r=0.0;
            }
          }

          if (spin==0){
            fprintf(fp_EV1,"%f %f ", T_kdis[kloop],kj_e*eV2Hartree);
          }
          else {
            fprintf(fp_EV3,"%f %f ",T_kdis[kloop],kj_e*eV2Hartree);
          }

          for (j=0; j<atomnum; j++) {
            if (spin==0) {
                for (k=0; k<Norbperatom[j]; k++){
                  fprintf(fp_EV1,"%e ",weight[j][k].r/coe);
                }
            }
            else {
              for (k=0; k<Norbperatom[j]; k++) {
                fprintf(fp_EV3,"%e ",weight[j][k].r/coe);
              } 
            } 
          }

          if (spin==0) {
            fprintf(fp_EV1,"\n");
          }
          else{
            fprintf(fp_EV3,"\n");
          } 
        } /* iloop */

      free(array_spin);
      free(array_i);

      //} /* if (myid==Host_ID) */

  }  /* kloop */

  if (fp_EV != NULL) {
    fd = fileno(fp_EV); 
    fsync(fd);
    fclose(fp_EV);
  }
  if (fp_EV1 != NULL){
    fd = fileno(fp_EV1); 
    fsync(fd);
    fclose(fp_EV1);
  }
  if ((SpinP_switch==1)&&(fp_EV2 != NULL)) {
    fd = fileno(fp_EV2); 
    fsync(fd);
    fclose(fp_EV2);
  }
  if ((SpinP_switch==1)&&(fp_EV3 != NULL)) {
    fd = fileno(fp_EV3); 
    fsync(fd);
    fclose(fp_EV3);
  }

  MPI_Barrier(mpi_comm_level1);

  /* Merge output files */

  if (myid==Host_ID){
    if (SpinP_switch==0) {

      sprintf(file_EV,".unfold_totup");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);

      sprintf(file_EV,".unfold_orbup");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);

    } 
    else if (SpinP_switch==1) {
      sprintf(file_EV,".unfold_totup");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);

      sprintf(file_EV,".unfold_orbup");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);

      sprintf(file_EV,".unfold_totdn");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);

      sprintf(file_EV,".unfold_orbdn");
      fnjoint(filepath,filename,file_EV);
      Merge_unfolding_output(file_EV, digit, numprocs);
    }
  }

  dtime(&Etime);


  /****************************************************
                       free arrays
  ****************************************************/

  free_atv_for_conceptual_cell();
  free_KGrids();
  free_kpath(); // set kpath and kname

  free(unfold_mapN2n);
  free(unfold_origin);
  free(np);
  free(num_mapn2N);
  free(count_k4S);
  free(K);

  for (j=0; j<natom; j++) free(mapn2N[j]);free(mapn2N);
  for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
  for (j=0; j<atomnum; j++) free(kj_v[j]); free(kj_v);
  for (i=0; i<Nthrds; i++) for (j=0; j<atomnum; j++) free(tmp_weight[i][j]);
  for (i=0; i<Nthrds; i++) free(tmp_weight[i]); free(tmp_weight);
  for (i=0; i<atomnum; i++) free(weight[i]); free(weight);

  free(Norbperatom);
  for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
  for (i=0; i<(unfold_Nkpoint+1); i++){
    free(unfold_kpoint_name[i]);
  }
  free(unfold_kpoint_name);

  /* free UnfoldBand parameters allocated in Input_std.c */
  if( UnfoldBand_path_flag == 1 ){
    free(UnfoldBand_N_perpath);
    for (i=0; i<(UnfoldBand_Nkpath+1); i++){
      for (j=0; j<3; j++){
        free(UnfoldBand_kpath[i][j]);
      }
      free(UnfoldBand_kpath[i]);
    }
    free(UnfoldBand_kpath);

    for (i=0; i<(UnfoldBand_Nkpath+1); i++){
      for (j=0; j<3; j++){
        free(UnfoldBand_kname[i][j]);
      }
      free(UnfoldBand_kname[i]);
    }
    free(UnfoldBand_kname);
  }

  free(array_kloopi);
  free(array_kloopj);

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);

  free(MP);
  free(order_GA);
  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (i=0; i<List_YOUSO[23]; i++){
    free(ko[i]);
  }
  free(ko);

  free(koS);

  for (j=0; j<List_YOUSO[23]; j++){
    free(EIGEN[j]);
  }
  free(EIGEN);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(H[i][j]);
    }
    free(H[i]);
  }
  free(H);  

  for (i=0; i<n+1; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(C[i][j]);
    }
    free(C[i]);
  }
  free(C);

  free(S1);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(H1[spin]);
  }
  free(H1);

  dtime(&TEtime);
  if (myid==Host_ID) printf("<Unfolding calculations end, time=%7.5f (s)>\n",TEtime-TStime);fflush(stdout);
}




static void Unfolding_Bands_NonCol(
                                   int SpinP_switch, 
                                   double *****nh,
                                   double *****ImNL,
                                   double ****CntOLP)
{

  double coe;
  double* a;
  double* b;
  double* c;
  double* K;
  double r[3];
  double r0[3];
  int rn, rn0;
  double kj_e;
  dcomplex** weight;
  dcomplex*** tmp_weight;
  //dcomplex** weight1;
  dcomplex** kj_v;
  dcomplex** kj_v1;
  double **fracabc;

  int i,j,k,l,n,m,jj1,jj2,n2;
  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int i1,j1,size_H1;
  int l1,l2,l3,kloop,AN,Rn;
  int h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN, wan1, TNO1, Anum;
  int Gh_AN, wan2, TNO2;
  int ii,MaxN;
  int mul,Gc_AN;
  int LB_AN,GB_AN,Bnum;

  double tmp,av_num;
  double k1,k2,k3;
  double *S1;
  double *RH0;
  double *RH1;
  double *RH2;
  double *RH3;
  double *IH0;
  double *IH1;
  double *IH2;
  double *ko,*M1,*EIGEN;
  double *koS;
  dcomplex **H,**S,**C;
  dcomplex Ctmp1,Ctmp2;
  double **Ctmp;
  double kRn,si,co;
  double TStime,TEtime,SiloopTime,EiloopTime;

  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV;
  FILE *fp_EV1;
  int numprocs,myid,ID,digit;
  int OMPID,Nthrds,Nprocs, index_thread;
  int *is1,*ie1;
  int *is2,*ie2;
  int *is12,*ie12;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  int boundary_id;
  int kloopi,kloopj;
  int* array_kloopi;
  int* array_kloopj;
  int iloop,total_iloop;
  int iloop_start, iloop_end;
  int* array_i;
  int* count_k4S;

  int MA,MB,MO,NO, mb;
  dcomplex dtmp,dtmp1;
  dcomplex phase1,phase2;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) {
    printf("\n*******************************************************\n");
    printf("                 Unfolding of Bands \n");
    printf("*******************************************************\n\n");fflush(stdout);
  } 

  dtime(&TStime);

  update_mapN2n(unfold_mapN2n);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;


  /****************************************************
   Allocation
  ****************************************************/

  getnorbperatom();

  set_nkpath();
  set_kpath();

  exitcode=0;

  set_atv_for_conceptual_cell(unfold_abc,unfold_origin,unfold_mapN2n);

  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    free(a);
    free(b);
    free(c);
    return;
  }

  coe=Cell_Volume/Cell_Volume_c;
  determine_kpts();

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);

  ko = (double*)malloc(sizeof(double)*n2);
  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double*)malloc(sizeof(double)*n2);

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  M1 = (double*)malloc(sizeof(double)*n2);

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    C[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  Ctmp = (double**)malloc(sizeof(double*)*n2);
  for (j=0; j<n2; j++){
    Ctmp[j] = (double*)malloc(sizeof(double)*n2);
  }

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  is12 = (int*)malloc(sizeof(int)*numprocs);
  ie12 = (int*)malloc(sizeof(int)*numprocs);

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  for (ID=0; ID<numprocs; ID++){
    is12[ID] = 2*is1[ID] - 1;
    ie12[ID] = 2*ie1[ID];
  }

  /* make is2 and ie2 */ 

  MaxN = 2*n;

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] =  1;
      ie2[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of arrays */
  S1  = (double*)malloc(sizeof(double)*size_H1);
  RH0 = (double*)malloc(sizeof(double)*size_H1);
  RH1 = (double*)malloc(sizeof(double)*size_H1);
  RH2 = (double*)malloc(sizeof(double)*size_H1);
  RH3 = (double*)malloc(sizeof(double)*size_H1);
  IH0 = (double*)malloc(sizeof(double)*size_H1);
  IH1 = (double*)malloc(sizeof(double)*size_H1);
  IH2 = (double*)malloc(sizeof(double)*size_H1);

  /* set S1, RH0, RH1, RH2, RH3, IH0, IH1, IH2 */

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1,  MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[0],  RH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[1],  RH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[2],  RH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[3],  RH3, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
      && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  
    
    /* nothing is done. */
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, ImNL[0], IH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[1], IH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[2], IH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  count_k4S = (int*)malloc(sizeof(int)*atomnum);
  k=0;
  for (MA=0; MA<atomnum; MA++) {
    count_k4S[MA] = k;
    GA_AN = order_GA[MA+1];
    wan1 = WhatSpecies[GA_AN];
    TNO1 = Spe_Total_CNO[wan1];
    for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++) {
      Gh_AN = natn[GA_AN][h_AN];
      wan2 = WhatSpecies[Gh_AN];
      TNO2 = Spe_Total_CNO[wan2];

      k = k + TNO1*TNO2;
    }
  }

  K=(double*)malloc(sizeof(double)*3);

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  set_KGrids();

#pragma omp parallel
  {
    Nthrds = omp_get_num_threads();
  }

  //printf("myid, Nthrds=%d, %d\n",myid,Nthrds);fflush(stdout);
  tmp_weight=(dcomplex***)malloc(sizeof(dcomplex**)*Nthrds);
  for (i=0; i<Nthrds; i++){
    tmp_weight[i]=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
    for (j=0; j<atomnum; j++){
      tmp_weight[i][j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
    }
  }


  weight=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (i=0; i<atomnum; i++) weight[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);
  //weight1=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  //for (i=0; i<atomnum; i++) weight1[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);


  kj_v=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (j=0; j<atomnum; j++) kj_v[j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
  kj_v1=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (j=0; j<atomnum; j++) kj_v1[j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);


  Name_Angular[0][0] = "s          ";
  Name_Angular[1][0] = "px         ";
  Name_Angular[1][1] = "py         ";
  Name_Angular[1][2] = "pz         ";
  Name_Angular[2][0] = "d3z^2-r^2  ";
  Name_Angular[2][1] = "dx^2-y^2   ";
  Name_Angular[2][2] = "dxy        ";
  Name_Angular[2][3] = "dxz        ";
  Name_Angular[2][4] = "dyz        ";
  Name_Angular[3][0] = "f5z^2-3r^2 ";
  Name_Angular[3][1] = "f5xz^2-xr^2";
  Name_Angular[3][2] = "f5yz^2-yr^2";
  Name_Angular[3][3] = "fzx^2-zy^2 ";
  Name_Angular[3][4] = "fxyz       ";
  Name_Angular[3][5] = "fx^3-3*xy^2";
  Name_Angular[3][6] = "f3yx^2-y^3 ";
  Name_Angular[4][0] = "g1         ";
  Name_Angular[4][1] = "g2         ";
  Name_Angular[4][2] = "g3         ";
  Name_Angular[4][3] = "g4         ";
  Name_Angular[4][4] = "g5         ";
  Name_Angular[4][5] = "g6         ";
  Name_Angular[4][6] = "g7         ";
  Name_Angular[4][7] = "g8         ";
  Name_Angular[4][8] = "g9         ";

  Name_Multiple[0] = "0";
  Name_Multiple[1] = "1";
  Name_Multiple[2] = "2";
  Name_Multiple[3] = "3";
  Name_Multiple[4] = "4";
  Name_Multiple[5] = "5";

  if (myid==Host_ID){
    strcpy(file_EV,".EV");
    fnjoint(filepath,filename,file_EV);
    if ((fp_EV = fopen(file_EV,"a")) != NULL){
      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"          Unfolding calculation for band structure         \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"                                                                          \n");
      fprintf(fp_EV," Origin of the Reference cell is set to (%f %f %f) (Bohr).\n\n",
              unfold_origin[0],unfold_origin[1],unfold_origin[2]);
      fprintf(fp_EV," Unfolded weights at specified k points are stored in System.Name.unfold_totup(dn).\n");
      fprintf(fp_EV," Individual orbital weights are stored in System.Name.unfold_orbup(dn).\n");
      fprintf(fp_EV," The format is: k_dis(Bohr^{-1})  energy(eV)  weight.\n\n");
      fprintf(fp_EV," The sequence for the orbital weights in System.Name.unfold_orbup(dn) is given below.\n\n");
      fflush(stdout);

      i1 = 1;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        wan1 = WhatSpecies[Gc_AN];
        for (l=0; l<=Supported_MaxL; l++){
          for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
            for (m=0; m<(2*l+1); m++){
              fprintf(fp_EV,"  %4d ",i1);
              if (l==0 && mul==0 && m==0)
                fprintf(fp_EV,"%4d %3s %s %s",
                        Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
              else
                fprintf(fp_EV,"         %s %s",
                        Name_Multiple[mul],Name_Angular[l][m]);
              fprintf(fp_EV,"\n");
              i1++;
            }
          }
        }
      }
      fflush(stdout);
  
      fprintf(fp_EV,"\n"); 
      fprintf(fp_EV,"\n  The total number of calculated k points is %i.\n",totnkpts);
      fprintf(fp_EV,"  The number of calculated k points on each path is \n");

      fprintf(fp_EV,"  For each path: ("); 
      for (i=1; i<=nkpath; i++){
        fprintf(fp_EV," %i",np[i]); 
      }
      fprintf(fp_EV," )\n\n");

      fprintf(fp_EV,"                 ka         kb         kc\n");
      fflush(stdout);

      for (kloop=1; kloop<=totnkpts; kloop++){
        fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);
      }

      fprintf(fp_EV,"\n");
      fclose(fp_EV);
    }
    else{
      printf("Failure of saving the EV file.\n");
      fclose(fp_EV);
    }
  }

  //strcpy(file_EV,".unfold_tot");
  sprintf(file_EV,".unfold_tot_%0*i",digit,myid);
  fnjoint(filepath,filename,file_EV);
  fp_EV = fopen(file_EV,"w");
  if (fp_EV == NULL) {
    printf("Failure of saving the System.Name.unfold_totup file.\n");
    fclose(fp_EV);
  }
  //strcpy(file_EV,".unfold_orb");
  sprintf(file_EV,".unfold_orb_%0*i",digit,myid);
  fnjoint(filepath,filename,file_EV);
  fp_EV1 = fopen(file_EV,"w");
  if (fp_EV1 == NULL) {
    printf("Failure of saving the System.Name.unfold_orbup file.\n");
    fclose(fp_EV1);
  }

  /* for gnuplot example */
  if (myid==Host_ID){
    generate_gnuplot_example(SpinP_switch);
  }
  /* end gnuplot example */

  /* for standard output */

  if (myid==Host_ID && 0<level_stdout) {
    printf(" The number of selected k points is %i.\n",totnkpts);

    printf(" For each path: (");
    for (i=1; i<=nkpath; i++){
      printf(" %i",np[i]); 
    }
    printf(" )\n\n");
    printf("                 ka         kb         kc\n"); fflush(stdout);
  }

  /*********************************************
                      kloopi 
  *********************************************/

  array_kloopi=(int*)malloc(sizeof(int)*(totnkpts+1));
  array_kloopj=(int*)malloc(sizeof(int)*(totnkpts+1));

  kloopi = 0;
  kloopj = 0;
  for (kloop=1; kloop<=totnkpts; kloop++) {
      array_kloopi[kloop] = kloopi;
      array_kloopj[kloop] = kloopj;
      //printf(kloop, array_kloopi[kloop], array_kloopj[kloop]);

      kloopj++;
      if(kloopj == np[kloopi]){
        kloopi++;
        kloopj = 0;
      }
  }

  /* MPI for kloop */
  //int kloop_start, kloop_end;
  //if (numprocs > totnkpts) {
  //  if (myid < numprocs){
  //    kloop_start = myid;
  //    kloop_end = kloop_start;
  //  }
  //  else {
  //      kloop_start = 0;
  //      kloop_end = -1;
  //  }
  //}
  //else {
  //  boundary_id_k = totnkpts%numprocs;
  //  if ( myid < boundary_id_k ){
  //    kloop_start = myid*(totnkpts/numprocs + 1);
  //    kloop_end = kloop_start + totnkpts/numprocs;
  //  }
  //  else {
  //    kloop_start = boundary_id_k*(totnkpts/numprocs + 1) + (myid - boundary_id_k)*(totnkpts/numprocs);
  //    kloop_end = kloop_start + totnkpts/numprocs - 1;
  //  }
  //}

  //printf("myid, kloop_start, kloop_end = %d %d %d\n", myid, kloop_start, kloop_end);fflush(stdout);



  //for (kloop=kloop_start; kloop<=kloop_end; kloop++)
  for (kloop=1; kloop<=totnkpts; kloop++) {
      kloopi = array_kloopi[kloop];
      kloopj = array_kloopj[kloop];

      /* for standard output */
     
      if (myid==Host_ID && 0<level_stdout) {

        if (kloop==totnkpts)
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);
        else 
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,T_kGrids1[kloop],T_kGrids2[kloop],T_kGrids3[kloop]);

        fflush(stdout);
      }

      k1 = T_KGrids1[kloop];
      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];

      /* make S and H */

      for (i=1; i<=n; i++){
        for (j=1; j<=n; j++){
          S[i][j] = Complex(0.0,0.0);
        } 
      } 

      for (i=1; i<=2*n; i++){
        for (j=1; j<=2*n; j++){
          H[i][j] = Complex(0.0,0.0);
        } 
      } 

      /* non-spin-orbit coupling and non-LDA+U */
      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
          && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  

        k = 0;
        for (AN=1; AN<=atomnum; AN++){
          GA_AN = order_GA[AN];
          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            GB_AN = natn[GA_AN][LB_AN];
            Rn = ncn[GA_AN][LB_AN];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            Bnum = MP[GB_AN];

            l1 = atv_ijk[Rn][1];
            l2 = atv_ijk[Rn][2];
            l3 = atv_ijk[Rn][3];
            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

            for (i=0; i<tnoA; i++){
              for (j=0; j<tnoB; j++){

                H[Anum+i  ][Bnum+j  ].r += co*RH0[k];
                H[Anum+i  ][Bnum+j  ].i += si*RH0[k];

                H[Anum+i+n][Bnum+j+n].r += co*RH1[k];
                H[Anum+i+n][Bnum+j+n].i += si*RH1[k];
            
                H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*RH3[k];
                H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*RH3[k];

                S[Anum+i  ][Bnum+j  ].r += co*S1[k];
                S[Anum+i  ][Bnum+j  ].i += si*S1[k];

                k++;
              }
            }
          }
        }
      }

      /* spin-orbit coupling or LDA+U */
      else {  

        k = 0;
        for (AN=1; AN<=atomnum; AN++){
          GA_AN = order_GA[AN];
          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
            GB_AN = natn[GA_AN][LB_AN];
            Rn = ncn[GA_AN][LB_AN];
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            Bnum = MP[GB_AN];

            l1 = atv_ijk[Rn][1];
            l2 = atv_ijk[Rn][2];
            l3 = atv_ijk[Rn][3];
            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

            for (i=0; i<tnoA; i++){
              for (j=0; j<tnoB; j++){

                H[Anum+i  ][Bnum+j  ].r += co*RH0[k] - si*IH0[k];
                H[Anum+i  ][Bnum+j  ].i += si*RH0[k] + co*IH0[k];

                H[Anum+i+n][Bnum+j+n].r += co*RH1[k] - si*IH1[k];
                H[Anum+i+n][Bnum+j+n].i += si*RH1[k] + co*IH1[k];
            
                H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*(RH3[k]+IH2[k]);
                H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*(RH3[k]+IH2[k]);

                S[Anum+i  ][Bnum+j  ].r += co*S1[k];
                S[Anum+i  ][Bnum+j  ].i += si*S1[k];

                k++;
              }
            }
          }
        }
      }

      /* set off-diagonal part */

      for (i=1; i<=n; i++){
        for (j=1; j<=n; j++){
          H[j+n][i].r = H[i][j+n].r;
          H[j+n][i].i =-H[i][j+n].i;
        } 
      } 

      /* diagonalization of S */
      Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

      /* minus eigenvalues to 1.0e-10 */

      for (l=1; l<=n; l++){
        if (koS[l]<1.0e-10) koS[l] = 1.0e-10;
      }

      /* calculate S*1/sqrt(koS) */

      for (l=1; l<=n; l++) koS[l] = 1.0/sqrt(koS[l]);

      /* S * 1.0/sqrt(koS[l]) */

#pragma omp parallel for shared(n,S,koS) private(i1,j1) 

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          S[i1][j1].r = S[i1][j1].r*koS[j1];
          S[i1][j1].i = S[i1][j1].i*koS[j1];
        } 
      } 

      /****************************************************
                  set H' and diagonalize it
      ****************************************************/

      /* U'^+ * H * U * M1 */

      /* transpose S */

      for (i1=1; i1<=n; i1++){
        for (j1=i1+1; j1<=n; j1++){
          Ctmp1 = S[i1][j1];
          Ctmp2 = S[j1][i1];
          S[i1][j1] = Ctmp2;
          S[j1][i1] = Ctmp1;
        }
      }

      /* H * U' */
      /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l)
      { 

        /* get info. on OpenMP */ 

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
          for (j1=is1[myid]; j1<=ie1[myid]; j1++){

            double sum_r0 = 0.0;
            double sum_i0 = 0.0;

            double sum_r1 = 0.0;
            double sum_i1 = 0.0;

            for (l=1; l<=n; l++){
              sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
              sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

              sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
              sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
            }

            C[2*j1-1][i1].r = sum_r0;
            C[2*j1-1][i1].i = sum_i0;

            C[2*j1  ][i1].r = sum_r1;
            C[2*j1  ][i1].i = sum_i1;
          }
        } 

      } /* #pragma omp parallel */

      /* U'^+ H * U' */
      /* H is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,jj1,jj2)
      { 

        /* get info. on OpenMP */ 

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (j1=is1[myid]+OMPID; j1<=ie1[myid]; j1+=Nthrds){
          for (i1=1; i1<=n; i1++){

            double sum_r00 = 0.0;
            double sum_i00 = 0.0;

            double sum_r01 = 0.0;
            double sum_i01 = 0.0;

            double sum_r10 = 0.0;
            double sum_i10 = 0.0;

            double sum_r11 = 0.0;
            double sum_i11 = 0.0;

            jj1 = 2*j1 - 1;
            jj2 = 2*j1;

            for (l=1; l<=n; l++){

              sum_r00 += S[i1][l].r*C[jj1][l  ].r + S[i1][l].i*C[jj1][l  ].i;
              sum_i00 += S[i1][l].r*C[jj1][l  ].i - S[i1][l].i*C[jj1][l  ].r;

              sum_r01 += S[i1][l].r*C[jj1][l+n].r + S[i1][l].i*C[jj1][l+n].i;
              sum_i01 += S[i1][l].r*C[jj1][l+n].i - S[i1][l].i*C[jj1][l+n].r;

              sum_r10 += S[i1][l].r*C[jj2][l  ].r + S[i1][l].i*C[jj2][l  ].i;
              sum_i10 += S[i1][l].r*C[jj2][l  ].i - S[i1][l].i*C[jj2][l  ].r;

              sum_r11 += S[i1][l].r*C[jj2][l+n].r + S[i1][l].i*C[jj2][l+n].i;
              sum_i11 += S[i1][l].r*C[jj2][l+n].i - S[i1][l].i*C[jj2][l+n].r;
            }

            H[jj1][2*i1-1].r = sum_r00;
            H[jj1][2*i1-1].i = sum_i00;

            H[jj1][2*i1  ].r = sum_r01;
            H[jj1][2*i1  ].i = sum_i01;

            H[jj2][2*i1-1].r = sum_r10;
            H[jj2][2*i1-1].i = sum_i10;

            H[jj2][2*i1  ].r = sum_r11;
            H[jj2][2*i1  ].i = sum_i11;

          }
        }

      } /* #pragma omp parallel */

      /* broadcast H */

      BroadCast_ComplexMatrix(mpi_comm_level1,H,2*n,is12,ie12,myid,numprocs,
                              stat_send,request_send,request_recv);

      /* H to C (transposition) */

#pragma omp parallel for shared(n,C,H)  

      for (i1=1; i1<=2*n; i1++){
        for (j1=1; j1<=2*n; j1++){
          C[j1][i1].r = H[i1][j1].r;
          C[j1][i1].i = H[i1][j1].i;
        }
      }

      /* solve the standard eigenvalue problem */
      /*  The output C matrix is distributed by column. */

      Eigen_PHH(mpi_comm_level1,C,ko,2*n,MaxN,0);

      for (i1=1; i1<=MaxN; i1++){
        EIGEN[i1] = ko[i1];
      }

      /* calculation of wave functions */

      /*  The H matrix is distributed by row */

      for (i1=1; i1<=2*n; i1++){
        for (j1=is2[myid]; j1<=ie2[myid]; j1++){
          H[j1][i1] = C[i1][j1];
        }
      }

      /* transpose */

      for (i1=1; i1<=n; i1++){
        for (j1=i1+1; j1<=n; j1++){
          Ctmp1 = S[i1][j1];
          Ctmp2 = S[j1][i1];
          S[i1][j1] = Ctmp2;
          S[j1][i1] = Ctmp1;
        }
      }

      /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is2,ie2,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,l1)
      { 

        /* get info. on OpenMP */ 

        OMPID = omp_get_thread_num();
        Nthrds = omp_get_num_threads();
        Nprocs = omp_get_num_procs();

        for (j1=is2[myid]+OMPID; j1<=ie2[myid]; j1+=Nthrds){
          for (i1=1; i1<=n; i1++){

            double sum_r0 = 0.0; 
            double sum_i0 = 0.0;

            double sum_r1 = 0.0; 
            double sum_i1 = 0.0;

            l1 = 0; 

            for (l=1; l<=n; l++){

              l1++; 

              sum_r0 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
              sum_i0 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;

              l1++; 

              sum_r1 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
              sum_i1 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;
            } 

            C[j1][i1  ].r = sum_r0;
            C[j1][i1  ].i = sum_i0;

            C[j1][i1+n].r = sum_r1;
            C[j1][i1+n].i = sum_i1;

          }
        }

      } /* #pragma omp parallel */

      /* broadcast C: C is distributed by row in each processor */

      BroadCast_ComplexMatrix(mpi_comm_level1,C,2*n,is2,ie2,myid,numprocs,
                              stat_send,request_send,request_recv);

      /* C to H (transposition)
         H consists of column vectors
      */ 

      //for (i1=1; i1<=MaxN; i1++){
      //  for (j1=1; j1<=2*n; j1++){
      //    H[j1][i1] = C[i1][j1];
      //  }
      //}

      MPI_Barrier(mpi_comm_level1);

      /****************************************************
                        Output
      ****************************************************/

      //if (myid==Host_ID){

        //setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

        K[0]=T_KGrids1[kloop]*rtv[1][1]+T_KGrids2[kloop]*rtv[2][1]+T_KGrids3[kloop]*rtv[3][1];
        K[1]=T_KGrids1[kloop]*rtv[1][2]+T_KGrids2[kloop]*rtv[2][2]+T_KGrids3[kloop]*rtv[3][2];
        K[2]=T_KGrids1[kloop]*rtv[1][3]+T_KGrids2[kloop]*rtv[2][3]+T_KGrids3[kloop]*rtv[3][3];

        iloop = 0;
        for (i=1; i<=2*n; i++){
          if (((EIGEN[i]-ChemP)<=unfold_ubound)&&((EIGEN[i]-ChemP)>=unfold_lbound)) {
            iloop++;
          }
        }
        total_iloop = iloop;

        array_i=(int*)malloc(sizeof(int)*total_iloop);

        iloop = 0;
        for (i=1; i<=2*n; i++){
          if (((EIGEN[i]-ChemP)<=unfold_ubound)&&((EIGEN[i]-ChemP)>=unfold_lbound)) {
            array_i[iloop] = i;
            iloop++;
          }
        }

        /* MPI for iloop */
        if (numprocs > total_iloop) {
          if (myid < total_iloop){
            iloop_start = myid;
            iloop_end = iloop_start;
          }
          else {
              iloop_start = 0;
              iloop_end = -1;
          }
        }
        else {
          boundary_id = total_iloop%numprocs;
          if ( myid < boundary_id ){
            iloop_start = myid*(total_iloop/numprocs + 1);
            iloop_end = iloop_start + total_iloop/numprocs;
          }
          else {
            iloop_start = boundary_id*(total_iloop/numprocs + 1) + (myid - boundary_id)*(total_iloop/numprocs);
            iloop_end = iloop_start + total_iloop/numprocs - 1;
          }
        }
        //printf("myid, iloop_start, iloop_end, total_iloop = %d, %d, %d, %d\n",myid, iloop_start, iloop_end, total_iloop);fflush(stdout);

        for (iloop=iloop_start; iloop<=iloop_end; iloop++){
          i = array_i[iloop];
          kj_e=(EIGEN[i]-ChemP);

          i1 = 1; 
          int iorb;

          for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
            iorb=0;

            wan1 = WhatSpecies[Gc_AN];
          
            for (l=0; l<=Supported_MaxL; l++){
              for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
                for (m=0; m<(2*l+1); m++){
                  kj_v[Gc_AN-1][iorb]=Complex(C[i][i1].r,C[i][i1].i);
                  kj_v1[Gc_AN-1][iorb]=Complex(C[i][i1+n].r,C[i][i1+n].i);

                  i1++;
                  iorb++;
                }
              }
            }
          }

          //printf("myid, Nthrds =%d\n",myid, Nthrds);
          for (l=0; l<Nthrds; l++) for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) tmp_weight[l][k][j]=Complex(0.,0.);
          for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight[k][j]=Complex(0.,0.);
          //for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight1[k][j]=Complex(0.,0.);


#pragma omp parallel for \
  shared( count_k4S, atomnum, order_GA, WhatSpecies, Spe_Total_CNO, FNAN,\
         natn, ncn, mapN2rn, num_mapn2N, mapn2N, a, b, c, K,\
         tmp_weight, unfold_mapN2n, kj_v, kj_v1, S1, myid, atv_c)\
  private( index_thread, k, MA, GA_AN, wan1, TNO1, h_AN, Gh_AN, Rn, wan2, TNO2,\
           r, rn, rn0, ii, l, MB, mb, r0, phase1, phase2, MO, NO, dtmp, dtmp1)
          for (MA=0; MA<atomnum; MA++) {
            index_thread = omp_get_thread_num();
            k = count_k4S[MA];
            GA_AN = order_GA[MA+1];
            wan1 = WhatSpecies[GA_AN];
            TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[GA_AN]; h_AN++) {
              Gh_AN = natn[GA_AN][h_AN];
              Rn = ncn[GA_AN][h_AN];
              wan2 = WhatSpecies[Gh_AN];
              TNO2 = Spe_Total_CNO[wan2];

              rn = mapN2rn[Gh_AN-1][Rn];
              r[0] = atv_c[rn][1];
              r[1] = atv_c[rn][2];
              r[2] = atv_c[rn][3];

              phase2=Cexp(Complex(0.,dot(K,r)));

              mb = unfold_mapN2n[Gh_AN-1];

              for (ii=0; ii<num_mapn2N[mb]; ii++){
                MB = mapn2N[mb][ii];
                rn0 = mapN2rn[MB][0];
                r0[0] = atv_c[rn0][1];
                r0[1] = atv_c[rn0][2];
                r0[2] = atv_c[rn0][3];

                phase1=Cmul(phase2,Cexp(Complex(0.,-dot(K,r0))));

                l=0;
                for (MO=0; MO<TNO1; MO++){
                  for (NO=0; NO<TNO2; NO++) {
                    dtmp=Cmul(Conjg(kj_v[GA_AN-1][MO]),kj_v[MB][NO]);
                    dtmp1=Cmul(Conjg(kj_v1[GA_AN-1][MO]),kj_v1[MB][NO]);
                    dtmp=Cadd(dtmp,dtmp1);
                    dtmp=RCmul(S1[k+l],dtmp);
                    //dtmp=RCmul(S1[k+l],tmpelem2[GA_AN-1][MB][MO][NO]);
                    //weight[Gh_AN-1][NO]=Cadd(weight[Gh_AN-1][NO],Cmul(phase2,dtmp));
                    dtmp=Cmul(phase1,dtmp);
                    tmp_weight[index_thread][MB][NO] = Cadd(tmp_weight[index_thread][MB][NO],dtmp);
                    l++;
                  }
                }
              }
              k = k + TNO1*TNO2;
            }
          }

          
          for (l=0; l<Nthrds; l++){
            for (j=0; j<atomnum; j++){
              for (k=0; k<Norbperatom[j]; k++){
                 weight[j][k] = Cadd(weight[j][k],tmp_weight[l][j][k]);
              }
            }
          }

          double sumallorb=0.;
          for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[j]; k++) sumallorb+=weight[j][k].r;
          fprintf(fp_EV,"%f %f %10.7f\n",T_kdis[kloop],kj_e*eV2Hartree,fabs(sumallorb)/coe);

          /* set negative weight to zero for plotting purpose */
          for (j=0; j<atomnum; j++){
            for (k=0; k<Norbperatom[j]; k++){

                if ( (weight[j][k].r)<0.0) {
                 weight[j][k].r  = 0.0; 
              }
            }
          }

          fprintf(fp_EV1,"%f %f ",T_kdis[kloop],kj_e*eV2Hartree);
          for (j=0; j<atomnum; j++) {
            for (k=0; k<Norbperatom[j]; k++) fprintf(fp_EV1,"%e ",(weight[j][k].r)/coe);
          }
          fprintf(fp_EV1,"\n");

        } /* iloop */  

        free(array_i);

      //} /* if (myid==Host_ID) */

      MPI_Barrier(mpi_comm_level1);

  }  /* kloop */


  if (fp_EV != NULL) fclose(fp_EV);
  if (fp_EV1 != NULL) fclose(fp_EV1);

  MPI_Barrier(mpi_comm_level1);

  /* Merge output files */

  if (myid==Host_ID){
    sprintf(file_EV,".unfold_tot");
    fnjoint(filepath,filename,file_EV);
    Merge_unfolding_output(file_EV, digit, numprocs);

    sprintf(file_EV,".unfold_orb");
    fnjoint(filepath,filename,file_EV);
    Merge_unfolding_output(file_EV, digit, numprocs);
  }


  /****************************************************
                       free arrays
  ****************************************************/

  free_atv_for_conceptual_cell();
  free_KGrids();
  free_kpath(); // set kpath and kname

  free(unfold_mapN2n);
  free(unfold_origin);
  free(np);
  free(num_mapn2N);
  free(count_k4S);
  free(K);

  for (j=0; j<natom; j++) free(mapn2N[j]);free(mapn2N);
  for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
  for (j=0; j<atomnum; j++) free(kj_v[j]); free(kj_v);
  for (j=0; j<atomnum; j++) free(kj_v1[j]); free(kj_v1);
  for (i=0; i<Nthrds; i++) for (j=0; j<atomnum; j++) free(tmp_weight[i][j]);
  for (i=0; i<Nthrds; i++) free(tmp_weight[i]); free(tmp_weight);
  for (i=0; i<atomnum; i++) free(weight[i]); free(weight);
  //for (i=0; i<atomnum; i++) free(weight1[i]); free(weight1);

  free(Norbperatom);
  for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
  for (i=0; i<(unfold_Nkpoint+1); i++){
    free(unfold_kpoint_name[i]);
  }
  free(unfold_kpoint_name);

  /* free UnfoldBand parameters allocated in Input_std.c */
  if( UnfoldBand_path_flag == 1 ){
    free(UnfoldBand_N_perpath);
    for (i=0; i<(UnfoldBand_Nkpath+1); i++){
      for (j=0; j<3; j++){
        free(UnfoldBand_kpath[i][j]);
      }
      free(UnfoldBand_kpath[i]);
    }
    free(UnfoldBand_kpath);

    for (i=0; i<(UnfoldBand_Nkpath+1); i++){
      for (j=0; j<3; j++){
        free(UnfoldBand_kname[i][j]);
      }
      free(UnfoldBand_kname[i]);
    }
    free(UnfoldBand_kname);
  }
  free(array_kloopi);
  free(array_kloopj);

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);
  free(is2);
  free(ie2);
  free(is12);
  free(ie12);

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(ko);
  free(koS);

  free(S1);
  free(RH0);
  free(RH1);
  free(RH2);
  free(RH3);
  free(IH0);
  free(IH1);
  free(IH2);

  free(EIGEN);

  for (j=0; j<n2; j++){
    free(H[j]);
  }
  free(H);

  for (i=0; i<n2; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (j=0; j<n2; j++){
    free(C[j]);
  }
  free(C);

  for (j=0; j<n2; j++){
    free(Ctmp[j]);
  }
  free(Ctmp);

  dtime(&TEtime);
  if (myid==Host_ID) printf("<Unfolding calculations end, time=%7.5f (s)>\n",TEtime-TStime);

}


static double volume(const double* a,const double* b,const double* c) {
  return fabs(a[0]*b[1]*c[2]+b[0]*c[1]*a[2]+c[0]*a[1]*b[2]-c[0]*b[1]*a[2]-a[1]*b[0]*c[2]-a[0]*c[1]*b[2]);}

void update_mapN2n(int* mapN2n) {
  /* Re-numbering atoms in conceptual cell from 1 to natom. */
  int i, j, k;
  int flag;
  int* tmp_mapN2n;
  
  //printf("mapN2n original\n");
  //for (i=0; i<atomnum; i++) {
  //     printf("%d %d\n",i,mapN2n[i]);fflush(stdout);
  //}

  tmp_mapN2n=(int*)malloc(sizeof(int)*atomnum);

  tmp_mapN2n[0] = 0;
  k=1;
  for (i=1; i<atomnum; i++) {
    flag=0;
    for (j=0; j<i; j++) {
      if(mapN2n[j]==mapN2n[i]){
        tmp_mapN2n[i] = tmp_mapN2n[j];
        flag = 1;
        break;
      }
    }
    if (flag == 0){
      tmp_mapN2n[i] = k;
      k++;
    }
  }

  natom = k;
  //printf("natom=%d\n",natom);

  for (i=0; i<atomnum; i++) {
    mapN2n[i] = tmp_mapN2n[i];
  }

  free(tmp_mapN2n);
  
  //printf("mapN2n\n");
  //for (i=0; i<atomnum; i++) {
  //     printf("%d %d\n",i,mapN2n[i]);fflush(stdout);
  //}

  num_mapn2N=(int*)malloc(sizeof(int)*natom);

  for (j=0; j<natom; j++) {
    k = 0;
    for (i=0; i<atomnum; i++) {
            if (j == mapN2n[i]){
                k++;
            }
    }
    num_mapn2N[j] = k;
  }
 
  mapn2N=(int**)malloc(sizeof(int*)*natom);
  for (j=0; j<natom; j++) {
    mapn2N[j]=(int*)malloc(sizeof(int)*num_mapn2N[j]);
  }

  for (j=0; j<natom; j++) {
    k = 0;
    for (i=0; i<atomnum; i++) {
      if (j == mapN2n[i]){
        mapn2N[j][k] = i;
        k++;
      }
    }
  }
  
  //printf("mapn2N\n");
  //printf("n, num_mapn2N, N\n");
  //for (j=0; j<natom; j++) {
  //  for (k=0; k<num_mapn2N[j]; k++) {
  //     printf("%d / %d, %d\n",j,num_mapn2N[j],mapn2N[j][k]);fflush(stdout);
  //  }
  //}
       

}

static double distwovec(const double* a, const double* b) {return sqrt((a[2]-b[2])*(a[2]-b[2])+(a[1]-b[1])*(a[1]-b[1])+(a[0]-b[0])*(a[0]-b[0]));}


/* abc = S ABC */
void abc_by_ABC(double ** S) { 
  double detABC=tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[1][2]*tv[2][1]*tv[3][3]-tv[1][1]*tv[3][2]*tv[2][3];
  int i,j,k;
  double** inv = (double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++) inv[i]=(double*)malloc(sizeof(double)*3); 
  inv[0][0]=(tv[2][2]*tv[3][3]-tv[3][2]*tv[2][3])/detABC;
  inv[0][1]=(tv[1][3]*tv[3][2]-tv[3][3]*tv[1][2])/detABC;
  inv[0][2]=(tv[1][2]*tv[2][3]-tv[2][2]*tv[1][3])/detABC;
  inv[1][0]=(tv[2][3]*tv[3][1]-tv[3][3]*tv[2][1])/detABC;
  inv[1][1]=(tv[1][1]*tv[3][3]-tv[3][1]*tv[1][3])/detABC;
  inv[1][2]=(tv[1][3]*tv[2][1]-tv[2][3]*tv[1][1])/detABC;
  inv[2][0]=(tv[2][1]*tv[3][2]-tv[3][1]*tv[2][2])/detABC;
  inv[2][1]=(tv[1][2]*tv[3][1]-tv[3][2]*tv[1][1])/detABC;
  inv[2][2]=(tv[1][1]*tv[2][2]-tv[2][1]*tv[1][2])/detABC;
  for (i=0; i<3; i++) for (j=0; j<3; j++) S[i][j]=0.;
  for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) S[i][j]+=unfold_abc[i][k]*inv[k][j];
  for (i=0; i<3; i++) free(inv[i]); free(inv);
}

static double dot(const double* v1,const double* v2) {
  double dotsum=0.;
  int i;
  for (i=0; i<3; i++) dotsum+=v1[i]*v2[i];
  return dotsum;
}

void getnorbperatom() {
  Norbperatom = (int*)malloc(sizeof(int)*atomnum);
  int ct_AN, wan1, TNO1;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    wan1 = WhatSpecies[ct_AN];
    TNO1 = Spe_Total_CNO[wan1];
    Norbperatom[ct_AN-1] = TNO1;
  }

  Norb=0;
  int* Ibegin;
  Ibegin = (int*)malloc(sizeof(int)*atomnum);
  int i;
  for (i=0; i<atomnum; i++) {Ibegin[i]=Norb; Norb+=Norbperatom[i];}
  free(Ibegin);
}



int set_tabr4RN(int*** tabr4RN,
                 double* auto_origin,
                 double* mergin
                 ) {

  int flag_shift_lattice;
  int max_tabr4RN;
  int i, j;
  double tmpxyz[4];
  int GA_AN;
  int Rn;
  double tmp[4],diff;
  int tmp_flag_shift_lattice[4];

  flag_shift_lattice = 0;
  max_tabr4RN = 0;

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    for (Rn=0; Rn<(TCpyCell+1); Rn++) {
      for (i=1; i<=3; i++){
        tmpxyz[i] = Gxyz[GA_AN][i] - auto_origin[i];
        for (j=1; j<=3; j++){
          tmpxyz[i] += atv_ijk[Rn][j]*tv[j][i];
        }
      }

      for (j=1; j<=3; j++){
        tmp_flag_shift_lattice[j] = 0;
   	    tmp[j] = Dot_Product(tmpxyz,rtv_c[j])*0.5/PI;
        if(tmp[j]<0.0){
          tabr4RN[GA_AN-1][Rn][j-1] = (int)tmp[j] - 1;
          diff = fabs(tmp[j] - (double)((int)tmp[j]) + 1.0); // 0 ~ 1
          if((diff < mergin[j]) || ((1.0-diff) < mergin[j])){
              tmp_flag_shift_lattice[j] = 1;
              //printf("j, GA_AN, Rn, tmp, (int)tmp, diff, mergin = %d %d %d %f %d %f %f\n", j, GA_AN, Rn, tmp[j], (int)tmp[j], diff, mergin[j]);fflush(stdout);
          }
        }
        else{
          tabr4RN[GA_AN-1][Rn][j-1] = (int)tmp[j];
          diff = fabs(tmp[j] - (double)((int)tmp[j]) ); // 0 ~ 1
          if((diff < mergin[j]) || ((1.0-diff) < mergin[j])){
              tmp_flag_shift_lattice[j] = 1;
              //printf("j, GA_AN, Rn, tmp, (int)tmp, diff, mergin = %d %d %d %f %d %f %f\n", j, GA_AN, Rn, tmp[j], (int)tmp[j], diff, mergin[j]);fflush(stdout);
          }
        }
      }

      if (3==(tmp_flag_shift_lattice[1]+tmp_flag_shift_lattice[2]+tmp_flag_shift_lattice[3])){
        flag_shift_lattice = 1;
      }

      //for (j=1; j<=3; j++){
      //  printf("GA_AN, Rn, j, tabr4RN = %d %d %d %d\n", GA_AN, Rn, j, tabr4RN[GA_AN-1][Rn][j-1]);fflush(stdout);
      //}

      for (j=1; j<=3; j++){
        if( max_tabr4RN < abs(tabr4RN[GA_AN-1][Rn][j-1]) ){
          max_tabr4RN = abs(tabr4RN[GA_AN-1][Rn][j-1]);
          //printf("GA_AN, Rn, j, max_tabr4RN, tabr4RN = %d %d %d %d %d\n", GA_AN, Rn, j, max_tabr4RN, tabr4RN[GA_AN-1][Rn][j-1]);fflush(stdout);
        }
      }
    }
  }

  CpyCell_c = max_tabr4RN;
  TCpyCell_c = (2*CpyCell_c+1)*(2*CpyCell_c+1)*(2*CpyCell_c+1);

  return flag_shift_lattice;

}

/* assign each R N with r */
void set_atv_for_conceptual_cell(double** lattice, double* origin, int* mapN2n) {
  /*** 
   Set the following valuables:
   tv_c, rtv_c
   auto_origin,
   atv_c[rn][3], atv_c_ijk[rn][3],
   ratv_c[i+CpyCell_c][j+CpyCell_c][k+CpyCell_c], 

   mapN2rn[GA_AN-1][Rn] = rn

   Cell_Volume_c
   
   * tabr4RN is not needed for later calculations.
   * Only mapN2rn, atv_c are needed for the unfolding weight calculation.
   
   ***/

  int i, j, k;
  double di,dj,dk;
  int l1,l2,l3,num;
  int GA_AN, ga_an;
  int Rn, rn, Rn2, rn2;
  int flag_shift_lattice;
  double tmp[4];
  double disp_origin[4],auto_origin[4],mergin[4];
  double CellV;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);


  /* lattice vector for a conceptual cell */
  tv_c[1][1] = lattice[0][0];
  tv_c[1][2] = lattice[0][1];
  tv_c[1][3] = lattice[0][2];
  tv_c[2][1] = lattice[1][0];
  tv_c[2][2] = lattice[1][1];
  tv_c[2][3] = lattice[1][2];
  tv_c[3][1] = lattice[2][0];
  tv_c[3][2] = lattice[2][1];
  tv_c[3][3] = lattice[2][2];

  Cross_Product(tv_c[2],tv_c[3],tmp);
  CellV = Dot_Product(tv_c[1],tmp); 
  Cell_Volume_c = CellV;

  /* reciprocal lattice vector for a conceptual cell */
  Cross_Product(tv_c[2],tv_c[3],tmp);
  rtv_c[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv_c[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv_c[1][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv_c[3],tv_c[1],tmp);
  rtv_c[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv_c[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv_c[2][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv_c[1],tv_c[2],tmp);
  rtv_c[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv_c[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv_c[3][3] = 2.0*PI*tmp[3]/CellV;

  /* Set the conceptual cell at the center of the supercell. */
  auto_origin[1] = (tv[1][1] + tv[2][1] + tv[3][1])*0.5;
  auto_origin[2] = (tv[1][2] + tv[2][2] + tv[3][2])*0.5;
  auto_origin[3] = (tv[1][3] + tv[2][3] + tv[3][3])*0.5;
  auto_origin[1] = auto_origin[1] - (tv_c[1][1] + tv_c[2][1] + tv_c[3][1])*0.5;
  auto_origin[2] = auto_origin[2] - (tv_c[1][2] + tv_c[2][2] + tv_c[3][2])*0.5;
  auto_origin[3] = auto_origin[3] - (tv_c[1][3] + tv_c[2][3] + tv_c[3][3])*0.5;


  /* to avoid atoms on lattice vector */
  disp_origin[0] = 0.1;
  for (i=1; i<=3; i++){
    disp_origin[i] = disp_origin[0];
    auto_origin[i] = auto_origin[i] - disp_origin[i];
  }

  /* Set tabr4RN for (GA_AN, Rn) -> atv_c_ijk[rn][3] */
  for (i=1; i<=3; i++){
    mergin[i] = Dot_Product(disp_origin,rtv_c[i])*0.5/PI *0.5;
  }

  tabr4RN=(int***)malloc(sizeof(int**)*atomnum);
  for (i=0; i<atomnum; i++) {
  tabr4RN[i]=(int**)malloc(sizeof(int*)*(TCpyCell+1));
    for (j=0; j<(TCpyCell+1); j++) {
      tabr4RN[i][j]=(int*)malloc(sizeof(int)*3);
    }
  }

  //printf("CpyCell, TCpyCell = %d %d\n",CpyCell, TCpyCell);


  if(flag_unfold_origin==1){
    for (i=0; i<3; i++){
      auto_origin[i+1] = origin[i];
    }
    flag_shift_lattice = set_tabr4RN(tabr4RN, auto_origin, mergin);
  }
  else{
    flag_shift_lattice = set_tabr4RN(tabr4RN, auto_origin, mergin);
    //MPI_Barrier(mpi_comm_level1);
    //MPI_Finalize();
    //exit(0);

    //if (myid==Host_ID){
    //  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    //    for (Rn=0; Rn<(TCpyCell+1); Rn++) {
    //        printf("GA_AN, Rn, i, j, k = %d %d %d %d %d \n",GA_AN, Rn, tabr4RN[GA_AN-1][Rn][0], tabr4RN[GA_AN-1][Rn][1], tabr4RN[GA_AN-1][Rn][2]);
    //    }
    //  }
    //}

    /* shift origin again to avoid atoms on lattice vector */
    if(flag_shift_lattice==1){
      for (i=1; i<=3; i++){
        auto_origin[i] = auto_origin[i] - disp_origin[i];
      }
    }

    if(flag_shift_lattice==1){
      flag_shift_lattice = set_tabr4RN(tabr4RN, auto_origin, mergin);
    }
    //MPI_Barrier(mpi_comm_level1);
    //MPI_Finalize();
    //exit(0);
  }

  if(flag_shift_lattice==1){
    if (myid==Host_ID){
      printf("Failed in set auto_origin.\n");
      printf("Cannot assign atoms in the reference cell properly! Could be due to more than one same atom in the reference cell!\n");
      printf("Check the input file, maybe the structure is highly disordered or you need to set the reference origin by yourself!\n\n");
      if(flag_unfold_origin==1){
        printf("unfold_origin = %f %f %f \n",origin[0],origin[1],origin[2]);
      }
      else{
        printf("auto_origin = %f %f %f \n",auto_origin[1],auto_origin[2],auto_origin[3]);
      }
    }
  }


  //if (myid==Host_ID){
  //  printf("CpyCell_c, TCpyCell_c = %d %d\n",CpyCell_c, TCpyCell_c);
  //}
  //MPI_Barrier(mpi_comm_level1);
  //MPI_Finalize();
  //exit(0);

  atv_c = (double**)malloc(sizeof(double*)*(TCpyCell_c+1));
  for (i=0; i<(TCpyCell_c+1); i++){
    atv_c[i] = (double*)malloc(sizeof(double)*4);
  }

  atv_c_ijk = (int**)malloc(sizeof(int*)*(TCpyCell_c+1));
  for (i=0; i<(TCpyCell_c+1); i++){
    atv_c_ijk[i] = (int*)malloc(sizeof(int)*4);
  }

  num = 2*CpyCell_c + 4;
  ratv_c = (int***)malloc(sizeof(int**)*num);
  for (i=0; i<num; i++){
    ratv_c[i] = (int**)malloc(sizeof(int*)*num);
    for (j=0; j<num; j++){
      ratv_c[i][j] = (int*)malloc(sizeof(int)*num);
    }
  }

  rn=1;
  di = -(CpyCell_c+1);
  for (i=-CpyCell_c; i<=CpyCell_c; i++){
    di = di + 1.0;
    dj = -(CpyCell_c+1);
    for (j=-CpyCell_c; j<=CpyCell_c; j++){
      dj = dj + 1.0;
      dk = -(CpyCell_c+1);
      for (k=-CpyCell_c; k<=CpyCell_c; k++){
        dk = dk + 1.0;
        if (i==0 && j==0 && k==0){
          atv_c[0][1] = 0.0;
          atv_c[0][2] = 0.0;
          atv_c[0][3] = 0.0;
          atv_c_ijk[0][1] = 0;
          atv_c_ijk[0][2] = 0;
          atv_c_ijk[0][3] = 0;
          ratv_c[i+CpyCell_c][j+CpyCell_c][k+CpyCell_c] = 0;
        }
        else{
          atv_c[rn][1] = di*tv_c[1][1] + dj*tv_c[2][1] + dk*tv_c[3][1];
          atv_c[rn][2] = di*tv_c[1][2] + dj*tv_c[2][2] + dk*tv_c[3][2];
          atv_c[rn][3] = di*tv_c[1][3] + dj*tv_c[2][3] + dk*tv_c[3][3];
          atv_c_ijk[rn][1] = i;
          atv_c_ijk[rn][2] = j;
          atv_c_ijk[rn][3] = k;
          ratv_c[i+CpyCell_c][j+CpyCell_c][k+CpyCell_c] = rn;
          rn = rn+1;
        }
      }
    }
  }

  mapN2rn=(int**)malloc(sizeof(int*)*atomnum);
  for (i=0; i<atomnum; i++) {
    mapN2rn[i]=(int*)malloc(sizeof(int)*(TCpyCell+1));
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    for (Rn=0; Rn<(TCpyCell+1); Rn++) {
      l1 = tabr4RN[GA_AN-1][Rn][0];
      l2 = tabr4RN[GA_AN-1][Rn][1];
      l3 = tabr4RN[GA_AN-1][Rn][2];
      rn = ratv_c[l1+CpyCell_c][l2+CpyCell_c][l3+CpyCell_c];
      ga_an = mapN2n[GA_AN-1]+1;
      mapN2rn[GA_AN-1][Rn] = rn;
      //printf("GA_AN, Rn, ga_an, rn = %d, %d, %d, %d\n",GA_AN, Rn, ga_an, rn);fflush(stdout);
    }
  }

  /* Check whether the reference cell is proper. */
  if (myid==Host_ID){
    for (j=0; j<natom; j++) {
      for (k=0; k<num_mapn2N[j]; k++) {
        GA_AN = mapn2N[j][k] + 1;
        for (Rn=0; Rn<(TCpyCell+1); Rn++) {
          rn = mapN2rn[GA_AN-1][Rn];
          for (i=k+1; i<num_mapn2N[j]; i++) {
            for (Rn2=Rn+1; Rn2<(TCpyCell+1); Rn2++) {
              rn2 = mapN2rn[GA_AN-1][Rn2];
              if(rn==rn2){
                printf("Cannot assign atoms in the reference cell properly! Could be due to more than one same atom in the reference cell!\n");fflush(stdout);
                printf("Check the input file, maybe the structure is highly disordered or you need to set the reference origin by yourself!\n\n");fflush(stdout);
                printf("GA_AN, Rn, Rn2, rn = %d %d %d %d\n",GA_AN, Rn, Rn2, rn);fflush(stdout);
                l1 = tabr4RN[GA_AN-1][Rn][0];
                l2 = tabr4RN[GA_AN-1][Rn][1];
                l3 = tabr4RN[GA_AN-1][Rn][2];
                printf("GA_AN, Rn, rn, l1, l2, l3= %d %d %d %d %d %d\n",GA_AN, Rn, rn, l1, l2, l3);fflush(stdout);
                l1 = tabr4RN[GA_AN-1][Rn2][0];
                l2 = tabr4RN[GA_AN-1][Rn2][1];
                l3 = tabr4RN[GA_AN-1][Rn2][2];
                printf("GA_AN, Rn2, rn2, l1, l2, l3= %d %d %d %d %d %d\n",GA_AN, Rn2, rn2, l1, l2, l3);fflush(stdout);
                exitcode = 1;
              } 
            }
          }
        }
      }
    }
  }

  /* free unnecessary arrays */

  for (i=0; i<atomnum; i++) for (j=0; j<(TCpyCell+1); j++) free(tabr4RN[i][j]);
  for (i=0; i<atomnum; i++) free(tabr4RN[i]); free(tabr4RN);

  for (i=0; i<(TCpyCell_c+1); i++){
    free(atv_c_ijk[i]);
  }
  free(atv_c_ijk);

  num = 2*CpyCell_c + 4;
  for (i=0; i<num; i++){
    for (j=0; j<num; j++){
      free(ratv_c[i][j]);
    }
    free(ratv_c[i]);
  }
  free(ratv_c);

  //MPI_Barrier(mpi_comm_level1);
  ////printf("Finish setting atv_c!\n");
  //MPI_Finalize();
  //exit(0);

}


void free_atv_for_conceptual_cell() {

  int i,j;
  int num;

  for (i=0; i<(TCpyCell_c+1); i++){
    free(atv_c[i]);
  }
  free(atv_c);

  for (i=0; i<atomnum; i++) {
    free(mapN2rn[i]);
  }
  free(mapN2rn);

  //for (i=0; i<(TCpyCell_c+1); i++){
  //  free(atv_c_ijk[i]);
  //}
  //free(atv_c_ijk);

  //num = 2*CpyCell_c + 4;
  //for (i=0; i<num; i++){
  //  for (j=0; j<num; j++){
  //    free(ratv_c[i][j]);
  //  }
  //  free(ratv_c[i]);
  //}
  //free(ratv_c);

}

void set_nkpath() {

  if( UnfoldBand_path_flag == 0 ){
    nkpath = unfold_Nkpoint - 1;
  }
  else if( UnfoldBand_path_flag == 1 ){
    nkpath = UnfoldBand_Nkpath;
  }

}

void set_kpath() {

  int i,j,k;

  kpath = (double***)malloc(sizeof(double**)*(nkpath+1));
  for (i=0; i<(nkpath+1); i++){
    kpath[i] = (double**)malloc(sizeof(double*)*3);
    for (j=0; j<3; j++){
      kpath[i][j] = (double*)malloc(sizeof(double)*4);
      for (k=0; k<4; k++) kpath[i][j][k] = 0.0;
    }
  }

  kname = (char***)malloc(sizeof(char**)*(nkpath+1));
  for (i=0; i<(nkpath+1); i++){
    kname[i] = (char**)malloc(sizeof(char*)*3);
    //for (j=0; j<3; j++){
    //  kname[i][j] = (char*)malloc(sizeof(char)*YOUSO10);
    //}
  }

  if( UnfoldBand_path_flag == 0 ){
    for (i=1; i<=nkpath; i++) {
      for (j=1; j<=3; j++) {
        kpath[i][1][j] = unfold_kpoint[i-1][j];
        kpath[i][2][j] = unfold_kpoint[i  ][j];
      } 
      kname[i][1] = unfold_kpoint_name[i-1];
      kname[i][2] = unfold_kpoint_name[i  ];
    } 
  }
  else if( UnfoldBand_path_flag == 1 ){
    for (i=1; i<=nkpath; i++) {
      for (j=1; j<=3; j++) {
        kpath[i][1][j] = UnfoldBand_kpath[i][1][j];
        kpath[i][2][j] = UnfoldBand_kpath[i][2][j];
      } 
      kname[i][1] = UnfoldBand_kname[i][1];
      kname[i][2] = UnfoldBand_kname[i][2];
    } 

  }


  //printf("nkpath=%d\n", nkpath);
  //for (i=1; i<=nkpath; i++) {
  //    printf("kpath (%s -> %s) : [%f, %f, %f] -> [%f, %f, %f]\n", kname[i][1], kname[i][2], kpath[i][1][1],kpath[i][1][2],kpath[i][1][3],kpath[i][2][1],kpath[i][2][2],kpath[i][2][3]);
  //}

}



void free_kpath() {

  int i,j;

  for (i=0; i<(nkpath+1); i++){
    for (j=0; j<3; j++){
      free(kpath[i][j]);
    }
    free(kpath[i]);
  }
  free(kpath);

  for (i=0; i<(nkpath+1); i++){
    free(kname[i]);
  }
  free(kname);

}


void determine_kpts() {

  int i,j;
    
  double dis=0.;

  for (i=1; i<=nkpath; i++) 
    dis+=sqrt(pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][1]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][1]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][1],2)+
              pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][2]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][2]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][2],2)+
              pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][3]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][3]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][3],2));

  np = (int*)malloc(sizeof(int)*(nkpath+1));
  np[0] = 0;

  if (unfold_nkpts<=nkpath) {
    for (i=1; i<=nkpath; i++) np[i]=2;
    totnkpts=nkpath+1;
  } 
  else {
    double intvl=dis/(unfold_nkpts-1);
    for (i=1; i<=nkpath; i++) {
      np[i]=
        (int)(sqrt(pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][1]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][1]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][1],2)+
                   pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][2]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][2]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][2],2)+
                   pow((kpath[i][2][1]-kpath[i][1][1])*rtv_c[1][3]+(kpath[i][2][2]-kpath[i][1][2])*rtv_c[2][3]+(kpath[i][2][3]-kpath[i][1][3])*rtv_c[3][3],2))/
              intvl);
      if (np[i]==0) np[i]=1;
      np[i] = np[i] + 1; /* for edge */
    }
    totnkpts=0;
    for (i=1; i<=nkpath; i++) totnkpts+=np[i];
  }

}


/* rtv_c = S rtv */
void rtv_c_by_rtv(double ** S) { 
  double detABC=rtv[1][1]*rtv[2][2]*rtv[3][3]+rtv[2][1]*rtv[3][2]*rtv[1][3]+rtv[3][1]*rtv[1][2]*rtv[2][3]-rtv[3][1]*rtv[2][2]*rtv[1][3]-rtv[1][2]*rtv[2][1]*rtv[3][3]-rtv[1][1]*rtv[3][2]*rtv[2][3];
  int i,j,k;
  double** inv = (double**)malloc(sizeof(double*)*4);
  for (i=0; i<4; i++) inv[i]=(double*)malloc(sizeof(double)*4); 
  inv[1][1]=(rtv[2][2]*rtv[3][3]-rtv[3][2]*rtv[2][3])/detABC;
  inv[1][2]=(rtv[1][3]*rtv[3][2]-rtv[3][3]*rtv[1][2])/detABC;
  inv[1][3]=(rtv[1][2]*rtv[2][3]-rtv[2][2]*rtv[1][3])/detABC;
  inv[2][1]=(rtv[2][3]*rtv[3][1]-rtv[3][3]*rtv[2][1])/detABC;
  inv[2][2]=(rtv[1][1]*rtv[3][3]-rtv[3][1]*rtv[1][3])/detABC;
  inv[2][3]=(rtv[1][3]*rtv[2][1]-rtv[2][3]*rtv[1][1])/detABC;
  inv[3][1]=(rtv[2][1]*rtv[3][2]-rtv[3][1]*rtv[2][2])/detABC;
  inv[3][2]=(rtv[1][2]*rtv[3][1]-rtv[3][2]*rtv[1][1])/detABC;
  inv[3][3]=(rtv[1][1]*rtv[2][2]-rtv[2][1]*rtv[1][2])/detABC;
  for (i=1; i<=3; i++) for (j=1; j<=3; j++) S[i][j]=0.;
  for (i=1; i<=3; i++) for (j=1; j<=3; j++) for (k=1; k<=3; k++) S[i][j]+=rtv_c[i][k]*inv[k][j];
  for (i=0; i<=3; i++) free(inv[i]); free(inv);
}


void set_KGrids() {
  /* 
    Set T_kGrids, T_kdis and T_kpath_kdis.

    vec_K = Kgrid * rtv
          = kgrid * S * rtv
          = kgrid * rtv_c
  */

  int i, kloop, kloopi, kloopj;
  double** frac_rtv_c;
  double kdis, dis2pk;
  double* vec_K;
  double* vec_K2;

  vec_K=(double*)malloc(sizeof(double)*3);
  vec_K2=(double*)malloc(sizeof(double)*3);

  frac_rtv_c = (double**)malloc(sizeof(double*)*4);
  for (i=0; i<4; i++) frac_rtv_c[i]=(double*)malloc(sizeof(double)*4); 

  rtv_c_by_rtv(frac_rtv_c); /* S: rtv_c = frac_rtv_c * rtv */

  /* k-grid for conceptual cell */
  T_kGrids1 = (double*)malloc(sizeof(double)*totnkpts+1);
  T_kGrids2 = (double*)malloc(sizeof(double)*totnkpts+1);
  T_kGrids3 = (double*)malloc(sizeof(double)*totnkpts+1);

  /* k-grid for supercell */
  T_KGrids1 = (double*)malloc(sizeof(double)*totnkpts+1);
  T_KGrids2 = (double*)malloc(sizeof(double)*totnkpts+1);
  T_KGrids3 = (double*)malloc(sizeof(double)*totnkpts+1);

  /* k-distance for x-axis of the band */
  T_kdis = (double*)malloc(sizeof(double)*totnkpts+1);
  T_kpath_kdis = (double**)malloc(sizeof(double*)*nkpath+1);
  for (kloopi=0; kloopi<=nkpath; kloopi++) {
    T_kpath_kdis[kloopi] = (double*)malloc(sizeof(double)*2);
  }

  kloop = 1;
  for (kloopi=1; kloopi<=nkpath; kloopi++) {
    for (kloopj=1; kloopj<=np[kloopi]; kloopj++) {
      T_kGrids1[kloop] = kpath[kloopi][1][1] + (kloopj-1)*(kpath[kloopi][2][1]-kpath[kloopi][1][1])/(np[kloopi]-1);
      T_kGrids2[kloop] = kpath[kloopi][1][2] + (kloopj-1)*(kpath[kloopi][2][2]-kpath[kloopi][1][2])/(np[kloopi]-1);
      T_kGrids3[kloop] = kpath[kloopi][1][3] + (kloopj-1)*(kpath[kloopi][2][3]-kpath[kloopi][1][3])/(np[kloopi]-1);
      kloop++;
    }
  }

  for (kloop=1; kloop<=totnkpts; kloop++){
    T_KGrids1[kloop] =  T_kGrids1[kloop]*frac_rtv_c[1][1]
                       +T_kGrids2[kloop]*frac_rtv_c[2][1]
                       +T_kGrids3[kloop]*frac_rtv_c[3][1];
    T_KGrids2[kloop] =  T_kGrids1[kloop]*frac_rtv_c[1][2]
                       +T_kGrids2[kloop]*frac_rtv_c[2][2]
                       +T_kGrids3[kloop]*frac_rtv_c[3][2];
    T_KGrids3[kloop] =  T_kGrids1[kloop]*frac_rtv_c[1][3]
                       +T_kGrids2[kloop]*frac_rtv_c[2][3]
                       +T_kGrids3[kloop]*frac_rtv_c[3][3];
  }

  kdis = 0.0;
  kloop = 1;
  for (kloopi=1; kloopi<=nkpath; kloopi++) {

    T_kpath_kdis[kloopi][1] = kdis;

    vec_K[0]=T_KGrids1[kloop]*rtv[1][1]+T_KGrids2[kloop]*rtv[2][1]+T_KGrids3[kloop]*rtv[3][1];
    vec_K[1]=T_KGrids1[kloop]*rtv[1][2]+T_KGrids2[kloop]*rtv[2][2]+T_KGrids3[kloop]*rtv[3][2];
    vec_K[2]=T_KGrids1[kloop]*rtv[1][3]+T_KGrids2[kloop]*rtv[2][3]+T_KGrids3[kloop]*rtv[3][3];
    kloop += np[kloopi] - 1;

    vec_K2[0]=T_KGrids1[kloop]*rtv[1][1]+T_KGrids2[kloop]*rtv[2][1]+T_KGrids3[kloop]*rtv[3][1];
    vec_K2[1]=T_KGrids1[kloop]*rtv[1][2]+T_KGrids2[kloop]*rtv[2][2]+T_KGrids3[kloop]*rtv[3][2];
    vec_K2[2]=T_KGrids1[kloop]*rtv[1][3]+T_KGrids2[kloop]*rtv[2][3]+T_KGrids3[kloop]*rtv[3][3];

    dis2pk=distwovec(vec_K,vec_K2);
    kdis+=dis2pk;
    T_kpath_kdis[kloopi][2] = kdis;
    kloop++;
  }

  kdis=0.0;
  T_kdis[1] = kdis;
  for (kloop=2; kloop<=totnkpts; kloop++){
    vec_K[0]=T_KGrids1[kloop-1]*rtv[1][1]+T_KGrids2[kloop-1]*rtv[2][1]+T_KGrids3[kloop-1]*rtv[3][1];
    vec_K[1]=T_KGrids1[kloop-1]*rtv[1][2]+T_KGrids2[kloop-1]*rtv[2][2]+T_KGrids3[kloop-1]*rtv[3][2];
    vec_K[2]=T_KGrids1[kloop-1]*rtv[1][3]+T_KGrids2[kloop-1]*rtv[2][3]+T_KGrids3[kloop-1]*rtv[3][3];
    vec_K2[0]=T_KGrids1[kloop]*rtv[1][1]+T_KGrids2[kloop]*rtv[2][1]+T_KGrids3[kloop]*rtv[3][1];
    vec_K2[1]=T_KGrids1[kloop]*rtv[1][2]+T_KGrids2[kloop]*rtv[2][2]+T_KGrids3[kloop]*rtv[3][2];
    vec_K2[2]=T_KGrids1[kloop]*rtv[1][3]+T_KGrids2[kloop]*rtv[2][3]+T_KGrids3[kloop]*rtv[3][3];
    dis2pk=distwovec(vec_K,vec_K2);
    kdis+=dis2pk;
    T_kdis[kloop] = kdis;
  }

  //kloop = 1;
  //for (kloopi=1; kloopi<=nkpath; kloopi++) {
  //  kloop += np[kloopi] - 1;
  //  printf("kloop, kloopi, T_kpath_kdis, T_kdis = %d %d %f %f\n",kloop, kloopi, T_kpath_kdis[kloopi][2], T_kdis[kloop]);fflush(stdout);
  //  kloop++;
  //}
  //for (kloop=1; kloop<=totnkpts; kloop++){
  //  printf("kloop, T_kdis = %d %f\n",kloop, T_kdis[kloop]);fflush(stdout);
  //}

  for (i=0; i<4; i++) free(frac_rtv_c[i]); free(frac_rtv_c);

  free(vec_K);
  free(vec_K2);

}


void free_KGrids() {
  int kloopi;

  /* k-grid for conceptual cell */
  free(T_kGrids1);
  free(T_kGrids2);
  free(T_kGrids3);

  /* k-grid for supercell */
  free(T_KGrids1);
  free(T_KGrids2);
  free(T_KGrids3);

  /* k-distance for x-axis of the band */

  free(T_kdis);

  for (kloopi=0; kloopi<=nkpath; kloopi++) {
    free(T_kpath_kdis[kloopi]);
  }
  free(T_kpath_kdis);

}

void Merge_unfolding_output(const char* file_EV, const int digit, const int numprocs) {

  char operate[500],operate1[500],operate2[500];
  int ID,c;
  FILE *fp1,*fp2;

  /* check whether the file exists, and if it exists, remove it. */

  sprintf(operate1,"%s",file_EV);
  fp1 = fopen(operate1, "r");   

  if (fp1!=NULL){
    fclose(fp1); 
    remove(operate1);
  }

  /* merge all the fraction files */

  for (ID=0; ID<numprocs; ID++){

    sprintf(operate1,"%s",file_EV);
    fp1 = fopen(operate1, "a");   
    fseek(fp1,0,SEEK_END);

    sprintf(operate2,"%s_%0*i",file_EV,digit,ID);
    fp2 = fopen(operate2,"r");

    if (fp2!=NULL){
      for (c=getc(fp2); c!=EOF; c=getc(fp2))  putc(c,fp1); 
      fclose(fp2); 
    }

    fclose(fp1); 
  }

   /* remove all the fraction files */
  for (ID=0; ID<numprocs; ID++){
    sprintf(operate,"%s_%0*i",file_EV,digit,ID);
    remove(operate);
  }

}

void generate_gnuplot_example(int SpinP_switch){

  char file_EV[YOUSO10];
  FILE *fp_EV0;
  int kloopi;

  /* for gnuplot example */

  strcpy(file_EV,".unfold_plotexample");
  fnjoint(filepath,filename,file_EV);
  fp_EV0 = fopen(file_EV,"w");
  if (fp_EV0 == NULL) {
    printf("Failure of saving the System.Name.unfold_plotexample file.\n");
    fclose(fp_EV0);
  }

  fprintf(fp_EV0,"set yrange [%f:%f]\n",unfold_lbound*eV2Hartree,unfold_ubound*eV2Hartree); 
  fprintf(fp_EV0,"set ylabel 'Energy (eV)'\n");

  //printf("UnfoldBand_path_flag = %d\n",UnfoldBand_path_flag);fflush(stdout);

  fprintf(fp_EV0,"set xtics(");
  fprintf(fp_EV0,"'%s' %f,",kname[1][1],T_kpath_kdis[1][1]); 
  for (kloopi=1; kloopi<nkpath; kloopi++) {
    if(strcmp(kname[kloopi+1][1],kname[kloopi][2])==0){
      fprintf(fp_EV0,"'%s' %f,",kname[kloopi][2],T_kpath_kdis[kloopi][2]); 
    }
    else{
      fprintf(fp_EV0,"'%s/%s' %f,",kname[kloopi+1][1],kname[kloopi][2],T_kpath_kdis[kloopi][2]); 
    }
  }
  fprintf(fp_EV0,"'%s' %f)\n",kname[nkpath][2],T_kpath_kdis[nkpath][2]); 


    fprintf(fp_EV0,"set xrange [0:%f]\n",T_kpath_kdis[nkpath][2]);
    fprintf(fp_EV0,"set arrow nohead from 0,0 to %f,0\n",T_kpath_kdis[nkpath][2]);

  for (kloopi=1; kloopi<nkpath; kloopi++) {
    fprintf(fp_EV0,"set arrow nohead from ");
    fprintf(fp_EV0,"%f,%f to %f,%f\n",T_kpath_kdis[kloopi][2],unfold_lbound*eV2Hartree,T_kpath_kdis[kloopi][2],unfold_ubound*eV2Hartree);
  }

  fprintf(fp_EV0,"set style circle radius 0\n");
  if (SpinP_switch==0 || SpinP_switch==1){
    fprintf(fp_EV0,"plot '%s.unfold_totup' using 1:2:($3)*0.05 notitle with circles lc rgb 'red'\n",filename);
  }
  else if(SpinP_switch==3){
    fprintf(fp_EV0,"plot '%s.unfold_tot' using 1:2:($3)*0.05 notitle with circles lc rgb 'red'\n",filename);
  }

  if (fp_EV0 != NULL) fclose(fp_EV0);

}


#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include<string.h> 
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

double Set_XMCD_penalty(int mode,double *pre_spin_moment,int SCF_iter)
{ 
  static int firsttime=1;
  double TStime,TEtime;
  int i,j,kl,n,m,Rn;
  int Mc_AN,Gc_AN,h_AN,k,Cwan,Gh_AN,Hwan,so;
  int tno0,tno1,tno2,i1,j1,p,ct_AN,spin;
  int fan,jg,kg,wakg,jg0,Mj_AN0,j0;
  int Mj_AN,num,size1,size2;
  int *Snd_OLP_Size,*Rcv_OLP_Size;
  int Original_Mc_AN,po; 
  double rcutA,rcutB,rcut,dmp;
  double rs,rs2,rs4,rs5,rs6,al;
  double fac2,fac,lx,ly,lz,r;
  double time1,time2,time3;
  double stime,etime;
  dcomplex ***NLH;
  double *tmp_array;
  double *tmp_array2;
  double Stime_atom,Etime_atom;
  double Total_S;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Total_XMCD_Penalty[4] ;
  double Partial_XMCD_Penalty[4] ;

  double penalty_value = 0.5/27.2114;  // 0.5 eV 
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;
  double Total_SpinSx0, Total_SpinSy0, Total_SpinSz0;
  int random0,random1;
  char operate[YOUSO10];

  FILE* fp1;


 
//  sprintf(operate,"%s%s",filepath,xanes_gs_file);
//  operate=C2H2-XANES0.lcao;

  dtime(&TStime);

  if ( MYID_MPI_COMM_WORLD==Host_ID && mode == 0 ){
  sprintf(operate,"%s%s",filepath,xanes_gs_file);
    if ((fp1 = fopen(operate,"rb")) != NULL){

      /* read parameters */      
      fseek(fp1,sizeof(int)*5 + sizeof(double)*1,SEEK_SET);
      fread(pre_spin_moment, sizeof(double), 3, fp1); 

      /* close file pointer */
      fclose(fp1);

//      random0=77;
//      random1=77;
//      Total_S=4.444092623/2 ;
//      Total_S=1;
//      pre_spin_moment[0] = Total_S*sin(random0/180.0*PI)*cos(random1/180.0*PI);
//      pre_spin_moment[1] = Total_S*sin(random0/180.0*PI)*sin(random1/180.0*PI);
//      pre_spin_moment[2] = Total_S*cos(random0/180.0*PI);

      printf("%f,%f,%f ASKASK\n",pre_spin_moment[0],pre_spin_moment[1],pre_spin_moment[2]);fflush(stdout);

    }
    else{
      printf("fail to access lcao file.");fflush(stdout); 
    }
  }
	MPI_Bcast(&pre_spin_moment[0], 3, MPI_DOUBLE, Host_ID, mpi_comm_level1);

/*-----------------------------------------------------------

1. for SCF = 0, Set:
   Total_SpinSx = pre_spin_moment[0].
   Total_SpinSy = pre_spin_moment[1].
   Total_SpinSz = pre_spin_moment[2].

2. effective potential operator: 

   SUM_(i,up)(j,up) ( penalty_value * ( Total_SpinSz - pre_spin_moment[2] ) * |i'><j'| * OLP(i,j) )
   +SUM_(i,dn)(j,dn) ( -penalty_value * ( Total_SpinSz - pre_spin_moment[2] ) * |i'><j'| * OLP(i,j) )

   SUM_(i,up)(j,dn) ( penalty_value * ( Total_SpinSx - pre_spin_moment[0] ) * |i'><j'| * OLP(i,j) )
   +SUM_(i,dn)(j,up) ( penalty_value * ( Total_SpinSx - pre_spin_moment[0] ) * |i'><j'| * OLP(i,j) )
 
   SUM_(i,up)(j,dn) ( i * penalty_value * ( Total_SpinSy - pre_spin_moment[1] ) * |i'><j'| * OLP(i,j) )
   +SUM_(i,dn)(j,up) ( -i * penalty_value * ( Total_SpinSy - pre_spin_moment[1] ) * |i'><j'| * OLP(i,j) )

3. add to HCH.

---------------------------------------------------------------*/

  OneD_Nloop = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }  

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN; 
      OneD2h_AN[OneD_Nloop] = h_AN; 
      OneD_Nloop++;
    }
  }

  Total_SpinSx0 = Total_SpinS *sin(Total_SpinAngle0)*cos(Total_SpinAngle1);    
  Total_SpinSy0 = Total_SpinS *sin(Total_SpinAngle0)*sin(Total_SpinAngle1);    
  Total_SpinSz0 = Total_SpinS *cos(Total_SpinAngle0);    

#pragma omp parallel shared(Matomnum,time_per_atom,HCH,iHCH,iHNL,F_CH_flag,List_YOUSO,Dis,SpinP_switch,Spe_Total_NO,OLP,Spe_VPS_List,VPS_j_dependency,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,SO_switch) 
{   
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,j,Gc_AN,Cwan,fan,jg,i1,j1,i;
    int Mj_AN,Hwan,k,kg,wakg,kl;
    int p,m,n,L,L1,L2,L3;
    double rcutA,rcutB,rcut,sum,ene;
    double Stime_atom, Etime_atom;
    double ene_m,ene_p,dmp;
    double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
    double PFp,PFm;

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    Partial_XMCD_Penalty[0]=0;
    Partial_XMCD_Penalty[1]=0;
    Partial_XMCD_Penalty[2]=0;


  for (Nloop = OMPID * OneD_Nloop / Nthrds ; Nloop < (OMPID+1) * OneD_Nloop / Nthrds; Nloop++){

    dtime(&Stime_atom);

    /* get Mc_AN and j */

    Mc_AN = OneD2Mc_AN[Nloop];
    j     = OneD2h_AN[Nloop];

    /* set data on Mc_AN */
    
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    fan = FNAN[Gc_AN];

    /* set data on j */
    
    jg = natn[Gc_AN][j];
    Mj_AN = F_G2M[jg];

    if (Mj_AN<=Matomnum){

    Hwan = WhatSpecies[jg]; 

    for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
      for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){


/*

   	 H_XMCD[0][Mc_AN][j][i1][j1]
   	   = + 10.0 * OLP[0][Mc_AN][j][i1][j1] /SCF_iter /SCF_iter /SCF_iter ;  // (up,up)

   	 H_XMCD[1][Mc_AN][j][i1][j1]
   	   = - 10.0 * OLP[0][Mc_AN][j][i1][j1] /SCF_iter /SCF_iter /SCF_iter ;  // (dn.dn)

     printf("%f	%f	%d\n",H_XMCD[1][Mc_AN][j][i1][j1],OLP[0][Mc_AN][j][i1][j1],SCF_iter);
*/



   if ( SCF_iter > 0 ){

   	 H_XMCD[0][Mc_AN][j][i1][j1]
   	   =penalty_value * (Total_SpinSz0 - pre_spin_moment[2]) * OLP[0][Mc_AN][j][i1][j1];  // (up,up)

   	 H_XMCD[1][Mc_AN][j][i1][j1]
   	   =-penalty_value * (Total_SpinSz0- pre_spin_moment[2]) * OLP[0][Mc_AN][j][i1][j1];  // (dn,dn)

     H_XMCD[2][Mc_AN][j][i1][j1] 
   	   =penalty_value *  (Total_SpinSx0- pre_spin_moment[0]) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 


   	 H_XMCD[3][Mc_AN][j][i1][j1] 
   	   =-penalty_value * (Total_SpinSy0- pre_spin_moment[1]) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 
     }
   else {
   	 H_XMCD[0][Mc_AN][j][i1][j1]
   	   =penalty_value * (- pre_spin_moment[2]) * OLP[0][Mc_AN][j][i1][j1];  // (up,up)

   	 H_XMCD[1][Mc_AN][j][i1][j1]
   	   =-penalty_value * (- pre_spin_moment[2]) * OLP[0][Mc_AN][j][i1][j1];  // (dn,dn)

   	 H_XMCD[2][Mc_AN][j][i1][j1] 
   	   =+penalty_value * (- pre_spin_moment[0]) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 

   	 H_XMCD[3][Mc_AN][j][i1][j1] 
   	   =-penalty_value * (- pre_spin_moment[1]) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 
    }

     Partial_XMCD_Penalty[0]+=H_XMCD[0][Mc_AN][j][i1][j1];
     Partial_XMCD_Penalty[1]+=H_XMCD[2][Mc_AN][j][i1][j1];
     Partial_XMCD_Penalty[2]+=H_XMCD[3][Mc_AN][j][i1][j1];


/*
   	 H_XMCD[0][Mc_AN][j][i1][j1]
   	   +=penalty_value * pow(Total_SpinSz0- pre_spin_moment[2],1) * OLP[0][Mc_AN][j][i1][j1];  // (up,up)
   	 H_XMCD[0][Mc_AN][j][i1][j1] = H_XMCD[0][Mc_AN][j][i1][j1]/2


   	 H_XMCD[1][Mc_AN][j][i1][j1]
   	   -=penalty_value * pow(Total_SpinSz0- pre_spin_moment[2],1) * OLP[0][Mc_AN][j][i1][j1];  // (dn,dn)
   	 H_XMCD[1][Mc_AN][j][i1][j1] = H_XMCD[1][Mc_AN][j][i1][j1]/2

     H_XMCD[2][Mc_AN][j][i1][j1] 
   	   +=penalty_value * pow(Total_SpinSx0- pre_spin_moment[0],1) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 
   	 H_XMCD[2][Mc_AN][j][i1][j1] = H_XMCD[2][Mc_AN][j][i1][j1]/2

   	 H_XMCD[3][Mc_AN][j][i1][j1] 
   	   -=penalty_value * pow(Total_SpinSy0- pre_spin_moment[1],1) * OLP[0][Mc_AN][j][i1][j1];  // (up,dn) 
     H_XMCD[3][Mc_AN][j][i1][j1] = H_XMCD[3][Mc_AN][j][i1][j1]/2


*/




//     printf("%f	%f	%f\n",pre_spin_moment[0]*2,pre_spin_moment[1]*2,pre_spin_moment[2]*2);

/*
       printf("%.10f	%.10f	%.10f	| %.10f\n", 
pre_spin_moment[0], pre_spin_moment[1], pre_spin_moment[2],
4*(pre_spin_moment[0]*pre_spin_moment[0]+pre_spin_moment[1]*pre_spin_moment[1]+pre_spin_moment[2]*pre_spin_moment[2]));

       printf("%.10f	%.10f	%.10f	| %.10f\n", Total_SpinSx0,Total_SpinSy0,Total_SpinSz0,4*(Total_SpinSx0*Total_SpinSx0+Total_SpinSy0*Total_SpinSy0+Total_SpinSz0*Total_SpinSz0));

       printf("%.10f	%.10f	%.10f	| %.10f\n", Total_SpinSx0-pre_spin_moment[0],Total_SpinSy0-pre_spin_moment[1],Total_SpinSz0-pre_spin_moment[2],
4*((Total_SpinSx0-pre_spin_moment[0])*(Total_SpinSx0-pre_spin_moment[0])+
 (Total_SpinSy0-pre_spin_moment[1])*(Total_SpinSy0-pre_spin_moment[1])+
 (Total_SpinSz0-pre_spin_moment[2])*(Total_SpinSz0-pre_spin_moment[2])));

	   printf("penalty =	%f	%f	%f	\n",H_XMCD[2][Mc_AN][j][i1][j1],H_XMCD[3][Mc_AN][j][i1][j1],H_XMCD[0][Mc_AN][j][i1][j1]);
       printf("Mc_AN: %d	j: %d	i1: %d	j1: %d\n\n", Mc_AN,j,i1,j1);
*/
      }
    }
    }
    }

#pragma omp barrier
#pragma omp flush(HNL,iHNL)

}
  
  MPI_Barrier(mpi_comm_level1);
  MPI_Allreduce(&Partial_XMCD_Penalty[0], &Total_XMCD_Penalty[0],1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
  MPI_Allreduce(&Partial_XMCD_Penalty[2], &Total_XMCD_Penalty[2],1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
  MPI_Allreduce(&Partial_XMCD_Penalty[3], &Total_XMCD_Penalty[3],1,MPI_DOUBLE,MPI_SUM,mpi_comm_level1);
  if ( MYID_MPI_COMM_WORLD==Host_ID ){
    printf("%f	%f	%f\n",Total_XMCD_Penalty[0],Total_XMCD_Penalty[2],Total_XMCD_Penalty[3]);
  }

  /* for time */
  dtime(&TEtime);
  return TEtime - TStime;

}

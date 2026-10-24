#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#ifndef STRESS_REAL_MPI
#define STRESS_REAL_MPI 0
#endif
enum {A=4,H=3,O=33};
static MPI_Comm mpi_comm_level1;
static int rank_id,rank_count,atomnum=3,Matomnum,SpeciesNum=3;
static int M2G[A],F_M2G[A],WhatSpecies[A]={0,0,1,2},Spe_Total_CNO[3]={1,2,3},Spe_Total_NO[3];
static int FNAN[A]={99,2,2,2},neighbors[A][H]={{0},{1,2,3},{2,3,1},{3,1,2}},*natn[A];
static int F_Snd_Num[4],F_Rcv_Num[4],F_TopMAN[4],snd_man[4][1],snd_gan[4][1],rcv_gan[4][1];
static int *Snd_MAN[4],*Snd_GAN[4],*Rcv_GAN[4];
#include "openmx_stress_workspace.h"
static void failure(int line,const char *what){
  fprintf(stderr,"FAIL line %d rank %d: %s\n",line,rank_id,what);
#if STRESS_REAL_MPI
  MPI_Abort(mpi_comm_level1,2);
#endif
  exit(2);
}
#define CHECK(c) do{if(!(c))failure(__LINE__,#c);}while(0)
#if !STRESS_REAL_MPI
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
int MPI_Comm_size(MPI_Comm comm,int *n){(void)comm;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm comm,int *r){(void)comm;*r=0;return 0;}
int MPI_Isend(const void *p,int n,int type,int dest,int tag,MPI_Comm comm,MPI_Request *r){(void)p;(void)n;(void)type;(void)dest;(void)tag;(void)comm;(void)r;CHECK(0);return 1;}
int MPI_Recv(void *p,int n,int type,int src,int tag,MPI_Comm comm,MPI_Status *s){(void)p;(void)n;(void)type;(void)src;(void)tag;(void)comm;(void)s;CHECK(0);return 1;}
int MPI_Wait(MPI_Request *r,MPI_Status *s){(void)r;(void)s;CHECK(0);return 1;}
#endif
/* Input matrices borrow fixed fixture rows. The changed production code
   owns only counts/messages, through the real invocation owner. */
static double values[4][A][H][O][O],*rows[4][A][H][O],**blocks[4][A][H],***atoms[4][A],****OLP[4];
static double seed(int component,int global,int neighbor,int i,int j){return 10000*component+1000*global+100*neighbor+2*i+.03125*j;}
#include "stress_overlap_native.inc"
static int callback(OpenMXSolverInvocation *owner,void *opaque,double *result){
  (void)opaque;int largest=0;
  for(int repeat=0;repeat<3;repeat++){
    Spe_Total_NO[0]=repeat==1?17:2;Spe_Total_NO[1]=repeat==1?24:1;Spe_Total_NO[2]=repeat==1?33:3;
    for(int c=0;c<4;c++)for(int slot=0;slot<A;slot++)for(int h=0;h<H;h++)for(int i=0;i<O;i++)for(int j=0;j<O;j++){
      const int global=F_M2G[slot];const int active=slot>0&&i<Spe_Total_NO[WhatSpecies[global]]&&j<Spe_Total_NO[WhatSpecies[neighbors[global][h]]];
      values[c][slot][h][i][j]=slot>0&&slot<=Matomnum&&active?seed(c,global,h,i,j):-98765;
    }
    int *send,*receive;OpenMX_Stress_OverlapCountsPrepare(owner,rank_count,&send,&receive);CHECK(send!=receive);
    double *send_buffer=OpenMX_Stress_HaloBuffer(owner,"stress:overlap:send",0);
    double *receive_buffer=OpenMX_Stress_HaloBuffer(owner,"stress:overlap:receive",0);
    CHECK(send_buffer&&receive_buffer&&send_buffer!=receive_buffer);
    MPI_OLP(owner,OLP);
    for(int c=0;c<4;c++)for(int slot=0;slot<A;slot++)for(int h=0;h<H;h++)for(int i=0;i<O;i++)for(int j=0;j<O;j++){
      const int global=F_M2G[slot];const int active=slot>0&&i<Spe_Total_NO[WhatSpecies[global]]&&j<Spe_Total_NO[WhatSpecies[neighbors[global][h]]];
      CHECK(values[c][slot][h][i][j]==(active?seed(c,global,h,i,j):-98765));
    }
    for(int peer=0;peer<rank_count;peer++){
      int want_send=0,want_receive=0;
      if(peer!=rank_id&&F_Snd_Num[peer]){
        const int g=snd_gan[peer][0];for(int h=0;h<H;h++)want_send+=4*Spe_Total_NO[WhatSpecies[g]]*Spe_Total_NO[WhatSpecies[neighbors[g][h]]];
      }
      if(peer!=rank_id&&F_Rcv_Num[peer]){
        const int g=rcv_gan[peer][0];for(int h=0;h<H;h++)want_receive+=4*Spe_Total_NO[WhatSpecies[g]]*Spe_Total_NO[WhatSpecies[neighbors[g][h]]];
      }
      CHECK(send[peer]==want_send&&receive[peer]==want_receive);if(want_send>largest)largest=want_send;if(want_receive>largest)largest=want_receive;
    }
    CHECK(FNAN[0]==99);
  }
  if(rank_count==4)CHECK((size_t)largest*sizeof(double)>65536);
  *result=1;return 1;
}
static void setup(void){
  CHECK(rank_count==1||rank_count==4);for(int g=0;g<A;g++)natn[g]=neighbors[g];
  if(rank_count==1){Matomnum=3;F_M2G[1]=M2G[1]=3;F_M2G[2]=M2G[2]=1;F_M2G[3]=M2G[3]=2;}
  else{
    Matomnum=rank_id<3?1:0;int next=Matomnum+1;if(Matomnum)F_M2G[1]=M2G[1]=rank_id+1;
    for(int peer=0;peer<4;peer++){
      Snd_MAN[peer]=snd_man[peer];Snd_GAN[peer]=snd_gan[peer];Rcv_GAN[peer]=rcv_gan[peer];if(peer==rank_id)continue;
      F_Snd_Num[peer]=Matomnum;snd_man[peer][0]=1;snd_gan[peer][0]=rank_id+1;
      if(peer<3){F_Rcv_Num[peer]=1;F_TopMAN[peer]=next;rcv_gan[peer][0]=peer+1;F_M2G[next++]=peer+1;}
    }CHECK(next==4);
  }
  for(int c=0;c<4;c++){OLP[c]=atoms[c];for(int g=0;g<A;g++){atoms[c][g]=blocks[c][g];for(int h=0;h<H;h++){blocks[c][g][h]=rows[c][g][h];for(int i=0;i<O;i++)rows[c][g][h][i]=values[c][g][h][i];}}}
}
int main(int argc,char **argv){
#if STRESS_REAL_MPI
  int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);CHECK(provided>=MPI_THREAD_FUNNELED);mpi_comm_level1=MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm_level1,&rank_id);MPI_Comm_size(mpi_comm_level1,&rank_count);
#else
  (void)argc;(void)argv;mpi_comm_level1=1;rank_count=1;
#endif
  setup();CHECK(OpenMX_Solver_Run(callback,NULL)==1);
  if(!rank_id)printf("PASS: overlap exchange, %d ranks, four components, nonidentity/empty ownership, unequal primitive/contracted orbitals, grow/shrink, %s\n",rank_count,rank_count==4?">64KiB messages":"no-peer path");
#if STRESS_REAL_MPI
  MPI_Finalize();
#endif
  return 0;
}

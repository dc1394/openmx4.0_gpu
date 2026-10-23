/* Exercise the actual private Stress.c producer with real invocation ownership.
   Inputs use fixed borrowed rows; no test allocation imitates output ownership. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_vna_matrix.h"
static MPI_Comm mpi_comm_level1=1;
int fixture_worker;
static void failure(int line,const char *what){fprintf(stderr,"FAIL line %d: %s\n",line,what);exit(2);}
#define CHECK(c) do{if(!(c))failure(__LINE__,#c);}while(0)
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
enum {A=5,H=3,O=4,P=8};
typedef float Type_DS_VNA;
static int atomnum=4,SpeciesNum=3,precision,Matomnum=2,List_YOUSO[40]={[34]=2,[35]=1};
static int M2G[A]={0,3,1},F_G2M[A]={0,2,3,1,4},WhatSpecies[A]={0,0,1,2,0};
static int Spe_Total_CNO[3]={1,2,3},FNAN[A]={99,2,2,2,2};
static int neighbors[A][H],images[A][H],rmi[A][H][H],*natn[A],*ncn[A],*rmi_rows[A][H],**RMI1[A];
static double coords[A][4]={{0},{0,.7,-.5,.2},{0,-.3,.8,1.1},{0,1.2,.1,-.6},{0,-.8,-.2,.4}},*Gxyz[A];
static double atv[H][4]={{0},{0,.17,-.11,.23},{0,-.13,.19,.07}};
static double distances[A][H],*Dis[A],Spe_Atom_Cut1[3]={4.1,4.3,4.7};
static float ds[4][A][H][O][P],*ds_rows[4][A][H][O],**ds_blocks[4][A][H],***ds_atoms[4][A],****DS[4];
static double hv[2][4][A][H][O][O],*hv_rows[2][4][A][H][O],**hv_blocks[2][4][A][H],***hv_atoms[2][4][A],****HV[2][4];
static double potential[A*H*O*O];
static size_t atom_offsets[A],block_offsets[A*H];static int shapes[A*H*2];
static OpenMXVnaMatrixViews vna;
#include "openmx_stress_workspace.h"
static int openmx_fp32_enabled(void){return precision;}
/* Smooth surrogate isolates damping terms and produces nonzero derivatives. */
static double dampingF(double cutoff,double r){return r>=cutoff?0:exp(-.07*r*r);}
static double deri_dampingF(double cutoff,double r){return r>=cutoff?0:-.14*r*dampingF(cutoff,r);}
#include "stress_vna_native.inc"
static int debug_h,debug_q,debug_where,debug_i,debug_j,debug_a,debug_b;
static void near(double a,double b){if(!isfinite(a)||fabs(a-b)>=3e-7*(1+fabs(b))){fprintf(stderr,"values %.17g %.17g h/q/where %d %d %d i/j/a/b %d %d %d %d\n",a,b,debug_h,debug_q,debug_where,debug_i,debug_j,debug_a,debug_b);CHECK(0);}}
static double product(int a,int b,int c,int i,int d,int e,int f,int j){
  double sum=0;for(int l=0;l<P;l++)sum+=precision?(double)(ds[a][b][c][i][l]*ds[d][e][f][j][l]):(double)ds[a][b][c][i][l]*(double)ds[d][e][f][j][l];
  return sum;
}
/* Scalar tensor algebra, deliberately independent of the output layout and
   production accumulation loops. Preserve two inherited stress scalings
   (symmetrization twice and damping twice); this ownership change does not
   adjudicate their physical interpretation. Full-app stress is compared too. */
static void reference(int where,int h,int q,int i,int j,double force[3],double stress[3][3]){
  const int g=M2G[1],ig=neighbors[g][h],jg=neighbors[g][q],mj=F_G2M[jg];
  const double damp=dampingF(Spe_Atom_Cut1[WhatSpecies[ig]]+Spe_Atom_Cut1[WhatSpecies[jg]],distances[ig][rmi[1][h][q]]);
  memset(force,0,3*sizeof(double));memset(stress,0,9*sizeof(double));
  if(!h&&!q&&!where){
    for(int k=1;k<H;k++)for(int a=0;a<3;a++){
      force[a]+=hv[0][a+1][1][k][i][j];
      for(int b=0;b<3;b++)stress[a][b]+=hv[0][a+1][1][k][i][j]*(coords[g][b+1]-coords[neighbors[g][k]][b+1]-atv[images[g][k]][b+1]);
    }
  }else if(h==q&&h){
    for(int a=0;a<3;a++)force[a]=-hv[1][a+1][1][h][i][j];
  }else if(!h){
    if(!where)for(int k=0;k<H;k++)if(rmi[1][q][k]>=0)for(int a=0;a<3;a++){
      const double value=product(a+1,1,k,i,0,mj<=Matomnum?mj:Matomnum+1,rmi[1][q][k],j);
      force[a]+=value;
      for(int b=0;b<3;b++)stress[a][b]+=value*(coords[g][b+1]-coords[neighbors[g][k]][b+1]-atv[images[g][k]][b+1]);
    }
    if(!q){
      for(int a=0;a<3;a++){
        double transposed=0;double s[3]={0};
        if(!where)for(int k=0;k<H;k++)if(rmi[1][0][k]>=0){
          const double value=product(a+1,1,k,j,0,F_G2M[g],rmi[1][0][k],i);transposed+=value;
          for(int b=0;b<3;b++)s[b]+=value*(coords[g][b+1]-coords[neighbors[g][k]][b+1]-atv[images[g][k]][b+1]);
        }
        force[a]+=transposed;for(int b=0;b<3;b++)stress[a][b]=2*(stress[a][b]+s[b]);
      }
    }else if(where){
      for(int a=0;a<3;a++)force[a]-=product(0,1,0,i,a+1,mj<=Matomnum?mj:Matomnum+1,rmi[1][mj<=Matomnum?q:0][mj<=Matomnum?0:q],j);
    }
  }else{
    for(int a=0;a<3;a++){
      force[a]=-product(a+1,Matomnum+1,rmi[1][0][h],i,0,Matomnum+1,rmi[1][0][q],j);
      if(q)force[a]-=product(0,Matomnum+1,rmi[1][0][h],i,a+1,Matomnum+1,rmi[1][0][q],j);
    }
  }
  for(int a=0;a<3;a++){force[a]*=damp;for(int b=0;b<3;b++)stress[a][b]*=damp*damp;}
  if((!h&&q)||(h&&!q)){
    const int k=h?h:q;double r=distances[g][k];
    const double ratio=r>=Spe_Atom_Cut1[WhatSpecies[ig]]+Spe_Atom_Cut1[WhatSpecies[jg]]?0:-.14*r*exp(-.07*r*r)/damp;
    const double p=h?potential[(H+k)*O*O+j*O+i]:potential[(H+k)*O*O+i*O+j];
    if(r<1e-10)r=1e-10;
    for(int a=0;a<3;a++){
      const double dx=coords[ig][a+1]+atv[images[g][h]][a+1]-coords[jg][a+1]-atv[images[g][q]][a+1];
      const double term=p*ratio*dx/r*(h?-1:1);force[a]+=term;
      if(!h)for(int b=0;b<3;b++)stress[a][b]+=term*(coords[ig][b+1]+atv[images[g][h]][b+1]-coords[jg][b+1]-atv[images[g][q]][b+1]);
    }
  }
}
/* Independent scalar energy for the on-center two-center contribution.
   Its derivative includes image translation and off-diagonal strain. */
static double strain_energy(int i,int j,int axis,int other,double epsilon){
  const int g=M2G[1];double value=0;
  for(int k=1;k<H;k++)for(int a=0;a<3;a++){
    const double r=coords[g][a+1]-coords[neighbors[g][k]][a+1]-atv[images[g][k]][a+1];
    const double rb=coords[g][other+1]-coords[neighbors[g][k]][other+1]-atv[images[g][k]][other+1];
    value+=hv[0][a+1][1][k][i][j]*(r+(a==axis?epsilon*rb:0));
  }return value;
}
static int callback(OpenMXSolverInvocation *owner,void *argument,double *result){
  const char *invalid=argument;int *list,*radial;OpenMXSolverRealTensor outputs[4],saved[4][4];int *saved_lists[4][2];
  if(invalid){
    if(!strcmp(invalid,"orbitals"))OpenMX_Stress_VnaPrepare(owner,0,&outputs[0],&outputs[1],&outputs[2],&outputs[3]);
    else OpenMX_Stress_VnaListsPrepare(owner,!strcmp(invalid,"radial")?-1:2,!strcmp(invalid,"angular")?-1:1,&list,&radial);
    CHECK(0);
  }
  OpenMX_Stress_VnaListsPrepare(owner,0,0,&list,&radial);CHECK(list&&radial&&list!=radial);list[1]=17;radial[1]=29;
  for(int repeat=0;repeat<3;repeat++){
    const int size=repeat==1?4:3;
    for(fixture_worker=0;fixture_worker<4;fixture_worker++){
      OpenMX_Stress_VnaListsPrepare(owner,repeat==1?3:2,1,&list,&radial);CHECK(list!=radial);saved_lists[fixture_worker][0]=list;saved_lists[fixture_worker][1]=radial;
      const int count=(repeat==1?3:2)*2+2;for(int k=0;k<count;k++){list[k]=100*fixture_worker+k;radial[k]=-100*fixture_worker-k;}
      OpenMX_Stress_VnaPrepare(owner,size,&outputs[0],&outputs[1],&outputs[2],&outputs[3]);
      for(int a=0;a<4;a++){CHECK(outputs[a].values);saved[fixture_worker][a]=outputs[a];for(int b=0;b<a;b++)CHECK(outputs[a].values!=outputs[b].values);}
      for(precision=0;precision<2;precision++)for(int remote=0;remote<2;remote++)for(int missing=0;missing<2;missing++){
        F_G2M[2]=remote?3:2;rmi[1][1][2]=missing?-1:2;
        for(int where=0;where<2;where++)for(int h=0;h<H;h++)for(int q=0;q<H;q++){
          if(rmi[1][h][q]<0)continue;
          debug_h=h;debug_q=q;debug_where=where;
          /* Poison all padding: only the active rectangle may be touched. */
          for(int a=0;a<3;a++)for(int k=0;k<size*size;k++)outputs[a].values[k]=12345;
          for(int k=0;k<16*size*size;k++)outputs[3].values[k]=12345;
          dHVNA(&vna,where,1,h,q,DS,HV[0],HV[1],&outputs[0],&outputs[1],&outputs[2],&outputs[3]);
          const int rows=Spe_Total_CNO[WhatSpecies[neighbors[M2G[1]][h]]],cols=Spe_Total_CNO[WhatSpecies[neighbors[M2G[1]][q]]];
          for(int i=0;i<size;i++)for(int j=0;j<size;j++){
            debug_i=i;debug_j=j;double f[3],s[3][3];if(i<rows&&j<cols)reference(where,h,q,i,j,f,s);
            for(int a=0;a<3;a++){debug_a=a;debug_b=-1;near(outputs[a].values[i*size+j],i<rows&&j<cols?f[a]:12345);}
            for(int a=0;a<4;a++)for(int b=0;b<4;b++){
              const double expected=i>=rows||j>=cols?12345:a&&b?s[a-1][b-1]:0;
              debug_a=a;debug_b=b;const double actual=outputs[3].values[((a*4+b)*size+i)*size+j];near(actual,expected);
              if(!where&&!h&&!q&&a&&b&&i<rows&&j<cols){const double delta=2e-6;near(actual,(strain_energy(i,j,a-1,b-1,delta)-strain_energy(i,j,a-1,b-1,-delta))/(2*delta));}
            }
          }
        }
      }
      for(int a=0;a<4;a++)outputs[a].values[0]=1000+fixture_worker*10+a;
    }
    /* Preparing later workers cannot invalidate earlier worker storage. */
    for(int w=0;w<4;w++)for(int k=0;k<(repeat==1?3:2)*2+2;k++){CHECK(saved_lists[w][0][k]==100*w+k);CHECK(saved_lists[w][1][k]==-100*w-k);}
    for(int w=0;w<4;w++)for(int a=0;a<4;a++)near(saved[w][a].values[0],1000+w*10+a);
  }
  fixture_worker=0;*result=1;return 1;
}
static void setup(void){
  for(int g=0;g<A;g++){
    natn[g]=neighbors[g];ncn[g]=images[g];Gxyz[g]=coords[g];Dis[g]=distances[g];RMI1[g]=rmi_rows[g];atom_offsets[g]=g*H;
    for(int h=0;h<H;h++){
      neighbors[g][h]=h==0?g:h==1?2:1;images[g][h]=h;rmi_rows[g][h]=rmi[g][h];
      for(int q=0;q<H;q++)rmi[g][h][q]=q;
      double d2=0;for(int a=1;a<=3;a++){const double d=coords[g][a]-coords[neighbors[g][h]][a]-atv[h][a];d2+=d*d;}distances[g][h]=sqrt(d2);
      const int b=g*H+h;block_offsets[b]=b*O*O;shapes[2*b]=shapes[2*b+1]=O;
    }
  }
  for(int k=0;k<A*H*O*O;k++)potential[k]=.2+.0007*k;
  vna.potential=(OpenMXDoubleBlockMatrix){{A,1,A*H,A*H*O*O,A*H*O*O,atom_offsets,block_offsets,shapes},potential};
  for(int c=0;c<4;c++){DS[c]=ds_atoms[c];for(int g=0;g<A;g++){
    ds_atoms[c][g]=ds_blocks[c][g];for(int h=0;h<H;h++){
      ds_blocks[c][g][h]=ds_rows[c][g][h];for(int i=0;i<O;i++){
        ds_rows[c][g][h][i]=ds[c][g][h][i];for(int p=0;p<P;p++)ds[c][g][h][i][p]=(float)(.07+.021*c-.013*g+.017*h+.011*i-.003*p);
      }
    }
  }}
  for(int t=0;t<2;t++)for(int c=0;c<4;c++){HV[t][c]=hv_atoms[t][c];for(int g=0;g<A;g++){
    hv_atoms[t][c][g]=hv_blocks[t][c][g];for(int h=0;h<H;h++){
      hv_blocks[t][c][g][h]=hv_rows[t][c][g][h];for(int i=0;i<O;i++){
        hv_rows[t][c][g][h][i]=hv[t][c][g][h][i];for(int j=0;j<O;j++)hv[t][c][g][h][i][j]=.031+.027*t-.023*c+.019*h+.011*i-.007*j;
      }
    }
  }}
}
int main(int argc,char **argv){setup();CHECK(OpenMX_Solver_Run(callback,argc>1?argv[1]:NULL)==1);puts("PASS: 1632 VNA branch cases across three capacities and four logical workers; rectangular padding and nine strain components");return 0;}

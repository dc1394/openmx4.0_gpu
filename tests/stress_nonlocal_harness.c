/* Borrowed fixture inputs; production C++ owns every output number. Native
   scalar projectors are checked by independent weighted bilinears. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_atomic.h"
#include "openmx_core_electronic.h"
enum {A=5,H=3,O=3,P=16};
static MPI_Comm mpi_comm_level1=1;
int fixture_worker;
static int atomnum=4,SpeciesNum=3,Matomnum=2,SpinP_switch,SO_switch,Hub_U_switch,F_U_flag;
static int Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch;
static int List_YOUSO[40]={[7]=3},M2G[A]={0,3,1},F_G2M[A]={0,2,3,1,4};
static int WhatSpecies[A]={0,0,1,2,0},Spe_Total_CNO[3]={1,2,3},FNAN[A]={99,2,2,2,2};
static int neighbors[A][H],images[A][H],rmi[A][H][H],*natn[A],*ncn[A],*rmi_rows[A][H],**RMI1[A];
static double coords[A][4]={{0},{0,.7,-.5,.2},{0,-.3,.8,1.1},{0,1.2,.1,-.6},{0,-.8,-.2,.4}},*Gxyz[A];
static double atv[H][4]={{0},{0,.17,-.11,.23},{0,-.13,.19,.07}},distances[A][H],*Dis[A];
static double cutoff[3]={4.1,4.3,4.7},energies[2][3][4];
static int radial[3],angular[3][5],projectors[3],relativistic[3];
static OpenMXAtomicViews atomic;
static OpenMXElectronicViews electronic;
static double ds[2][4][A][H][O][P],*ds_rows[2][4][A][H][O],**ds_blocks[2][4][A][H],***ds_atoms[2][4][A],****ds_parts[2][4],*****DS[2];
static double potential[2][3][A][H][O][O],*potential_rows[2][3][A][H][O],**potential_blocks[2][3][A][H],***potential_atoms[2][3][A],****HV[2][3];
#include "openmx_stress_workspace.h"
static void failure(int line,const char *what){fprintf(stderr,"FAIL line %d: %s\n",line,what);exit(2);}
#define CHECK(c) do{if(!(c))failure(__LINE__,#c);}while(0)
int MPI_Abort(MPI_Comm comm,int code){(void)comm;(void)code;_Exit(77);}
static dcomplex Complex(double r,double i){return (dcomplex){r,i};}
static double dampingF(double cutoff,double r){return r>=cutoff?0:exp(-.07*r*r);}
static double deri_dampingF(double cutoff,double r){return r>=cutoff?0:-.14*r*exp(-.07*r*r);}
#include "stress_nonlocal_native.inc"
static void near(double a,double b){if(!isfinite(a)||fabs(a-b)>3e-9*(1+fabs(b))){fprintf(stderr,"values %.17g %.17g\n",a,b);CHECK(0);}}
static double alpha(int l){return .08+.003*l;}
static double coefficient(int slot,int orbital,int l){return .17+.021*slot+.011*orbital-.003*l;}
static double energy_weight(int species,int l){
  int offset=0;for(int r=0;r<radial[species];r++){
    const int width=2*angular[species][r+1]+1;
    if(l<offset+width)return energies[0][species][r];offset+=width;
  }return 0;
}
static double bilinear(int species,int direction,int slot,int neighbor,int i,int slot2,int neighbor2,int j){
  double value=0;for(int l=0;l<P;l++)value+=energy_weight(species,l)*ds[0][direction][slot][neighbor][i][l]*ds[0][0][slot2][neighbor2][j][l];return value;
}
static void scalar_reference(int q,int i,int j,double f[3],double s[3][3]){
  const int g=M2G[1],jg=neighbors[g][q],mj=F_G2M[jg],target=mj<=Matomnum?mj:Matomnum+1;
  const double damp=dampingF(cutoff[WhatSpecies[g]]+cutoff[WhatSpecies[jg]],distances[g][q]);
  memset(f,0,3*sizeof(double));memset(s,0,9*sizeof(double));
  for(int k=0;k<H;k++)if(rmi[1][q][k]>=0)for(int a=0;a<3;a++){
    const int species=WhatSpecies[neighbors[g][k]];
    double value=bilinear(species,a+1,1,k,i,target,rmi[1][q][k],j);
    if(!q)value+=bilinear(species,a+1,1,k,j,1,k,i);
    f[a]+=value*damp;
    for(int b=0;b<3;b++)s[a][b]+=value*damp*(coords[g][b+1]-coords[neighbors[g][k]][b+1]-atv[images[g][k]][b+1]);
  }
}
static double strain_energy(int i,int j,int axis,int other,double epsilon){
  const int g=M2G[1];double value=0;
  for(int k=0;k<H;k++){
    double r[3],r2=0;for(int a=0;a<3;a++)r[a]=coords[g][a+1]-coords[neighbors[g][k]][a+1]-atv[images[g][k]][a+1];
    r[axis]+=epsilon*r[other];for(int a=0;a<3;a++)r2+=r[a]*r[a];
    for(int l=0;l<P;l++)value+=energy_weight(WhatSpecies[neighbors[g][k]],l)*coefficient(1,i,l)*coefficient(1,j,l)*exp(-2*alpha(l)*r2);
  }return value;
}
static double layout_value(int spin,int axis,int other,int i,int j){return .13+.19*spin-.17*axis+.11*other+.07*i-.03*j;}
static int comparisons;
static int callback(OpenMXSolverInvocation *owner,void *argument,double *result){
  OpenMXSolverComplexTensor view[7],saved[4][7];
  if(argument){OpenMX_Stress_HnlPrepare(owner,atoi(argument),&view[0],&view[1],&view[2],&view[3],&view[4],&view[5],&view[6]);CHECK(0);}
  for(int repeat=0;repeat<3;repeat++){
    const int size=repeat==1?5:3;List_YOUSO[7]=size;
    for(fixture_worker=0;fixture_worker<4;fixture_worker++){
      OpenMX_Stress_HnlPrepare(owner,size,&view[0],&view[1],&view[2],&view[3],&view[4],&view[5],&view[6]);
      for(int f=0;f<7;f++){saved[fixture_worker][f]=view[f];CHECK(view[f].values);for(int other=0;other<f;other++)CHECK(view[f].values!=view[other].values);}
      /* Direct five-index reference establishes the folded complex layout,
         independently of the accessor formula and native scalar producer. */
      dcomplex expected[3][4][4][5][5];
      for(int spin=0;spin<3;spin++)for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int i=0;i<size;i++)for(int j=0;j<size;j++){
        const double v=layout_value(spin,a,b,i,j);expected[spin][a][b][i][j]=Complex(v,-.37*v);
        OpenMX_Stress_HnlStrainRow(&view[6],spin,a,b,i)[j]=expected[spin][a][b][i][j];
      }
      for(int spin=0;spin<3;spin++)for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int i=0;i<size;i++)for(int j=0;j<size;j++){
        const dcomplex actual=view[6].values[(((spin*4+a)*4+b)*size+i)*size+j];near(actual.r,expected[spin][a][b][i][j].r);near(actual.i,expected[spin][a][b][i][j].i);
      }
      for(int mode=0;mode<3;mode++)for(int sign=-1;sign<=1;sign++)for(int sp=0;sp<3;sp++)for(int remote=0;remote<2;remote++)for(int missing=0;missing<2;missing++){
        SpinP_switch=sp==2?3:sp;F_G2M[2]=remote?3:2;rmi[1][1][2]=missing?-1:2;
        for(int w=0;w<3;w++){
          radial[w]=mode==0?0:mode==1?1:4;projectors[w]=mode==0?0:mode==1?1:16;
          for(int l=0;l<4;l++)for(int c=0;c<2;c++)energies[c][w][l]=sign*(.27+.13*w-.07*l+.09*c);
        }
        for(int q=0;q<H;q++){
          for(int f=0;f<7;f++){const int n=f==6?48*size*size:3*size*size;for(int k=0;k<n;k++)view[f].values[k]=Complex(12345,-7890);}
          dHNL(&atomic,0,1,0,q,DS,&view[0],&view[1],&view[2],&view[6],&electronic);
          const int rows=Spe_Total_CNO[WhatSpecies[M2G[1]]],cols=Spe_Total_CNO[WhatSpecies[neighbors[M2G[1]][q]]];
          for(int i=0;i<size;i++)for(int j=0;j<size;j++){
            double force[3]={0},strain[3][3]={{0}};if(i<rows&&j<cols)scalar_reference(q,i,j,force,strain);
            for(int spin=0;spin<3;spin++)for(int a=0;a<3;a++){
              double want=i<rows&&j<cols&&spin<2?force[a]:0;
              if(q&&i<rows&&j<cols&&spin<=(SpinP_switch==0?0:SpinP_switch==1?1:2)){
                const int g=M2G[1],jg=neighbors[g][q];const double r=distances[g][q],damp=dampingF(cutoff[WhatSpecies[g]]+cutoff[WhatSpecies[jg]],r);
                const double dx=-.14*r*exp(-.07*r*r)/damp*(coords[g][a+1]-coords[jg][a+1]-atv[q][a+1])/r;
                want+=potential[0][spin][1][q][i][j]*dx;
              }
              const dcomplex actual=view[a].values[(spin*size+i)*size+j];near(actual.r,want);near(actual.i,0);
            }
            for(int spin=0;spin<3;spin++)for(int a=0;a<4;a++)for(int b=0;b<4;b++){
              double want=i<rows&&j<cols&&spin<2&&a&&b?strain[a-1][b-1]:0;
              if(q&&i<rows&&j<cols&&a&&b&&spin<=(SpinP_switch==0?0:SpinP_switch==1?1:2)){
                const int g=M2G[1],jg=neighbors[g][q];const double r=distances[g][q],damp=dampingF(cutoff[WhatSpecies[g]]+cutoff[WhatSpecies[jg]],r);
                const double dx=-.14*r*exp(-.07*r*r)/damp*(coords[g][a]-coords[jg][a]-atv[q][a])/r;
                want+=potential[0][spin][1][q][i][j]*dx*(coords[g][b]-coords[jg][b]-atv[q][b]);
              }
              const dcomplex actual=view[6].values[(((spin*4+a)*4+b)*size+i)*size+j];near(actual.r,want);near(actual.i,0);
              if(!q&&i<rows&&j<cols&&spin<2&&a&&b){const double delta=2e-6;near(actual.r,(strain_energy(i,j,a-1,b-1,delta)-strain_energy(i,j,a-1,b-1,-delta))/(2*delta));}
            }
          }
          for(int f=3;f<6;f++)for(int k=0;k<3*size*size;k++){near(view[f].values[k].r,12345);near(view[f].values[k].i,-7890);}
          ++comparisons;
        }
      }
      for(int f=0;f<7;f++)view[f].values[0]=Complex(1000+10*fixture_worker+f,-fixture_worker-f);
    }
    for(int w=0;w<4;w++)for(int f=0;f<7;f++){near(saved[w][f].values[0].r,1000+10*w+f);near(saved[w][f].values[0].i,-w-f);}
  }
  fixture_worker=0;*result=1;return 1;
}
static void setup(void){
  for(int g=0;g<A;g++){
    natn[g]=neighbors[g];ncn[g]=images[g];Gxyz[g]=coords[g];Dis[g]=distances[g];RMI1[g]=rmi_rows[g];
    for(int h=0;h<H;h++){
      neighbors[g][h]=h==0?g:h==1?2:1;images[g][h]=h;rmi_rows[g][h]=rmi[g][h];for(int q=0;q<H;q++)rmi[g][h][q]=q;
      double r2=0;for(int a=1;a<=3;a++){const double d=coords[g][a]-coords[neighbors[g][h]][a]-atv[h][a];r2+=d*d;}distances[g][h]=sqrt(r2);
    }
  }
  atomic.species.species=3;atomic.species.contracted_orbitals=Spe_Total_CNO;atomic.species.cutoff=cutoff;atomic.species.radial_projectors=radial;
  atomic.species.projector_components=projectors;atomic.species.relativistic=relativistic;atomic.projector_angular=&angular[0][0];atomic.projector_angular_species=3;atomic.projector_angular_stride=5;
  atomic.projector_energies=(OpenMXRadialView3){&energies[0][0][0],{2,3,4}};for(int w=0;w<3;w++)for(int l=1;l<=4;l++)angular[w][l]=l-1;
  for(int c=0;c<2;c++){DS[c]=ds_parts[c];for(int a=0;a<4;a++){ds_parts[c][a]=ds_atoms[c][a];for(int g=0;g<A;g++){
    ds_atoms[c][a][g]=ds_blocks[c][a][g];for(int h=0;h<H;h++){ds_blocks[c][a][g][h]=ds_rows[c][a][g][h];for(int i=0;i<O;i++){
      ds_rows[c][a][g][h][i]=ds[c][a][g][h][i];const int center=g==1?3:g==2?1:2;double r[3],r2=0;
      for(int b=0;b<3;b++){r[b]=coords[center][b+1]-coords[neighbors[center][h]][b+1]-atv[h][b+1];r2+=r[b]*r[b];}
      for(int l=0;l<P;l++){const double v=coefficient(g,i,l)*exp(-alpha(l)*r2);ds[c][a][g][h][i][l]=(c?.31:1)*(a?-2*alpha(l)*r[a-1]*v:v);}
    }}
  }}}
  for(int c=0;c<2;c++)for(int spin=0;spin<3;spin++){HV[c][spin]=potential_atoms[c][spin];for(int g=0;g<A;g++){
    potential_atoms[c][spin][g]=potential_blocks[c][spin][g];for(int h=0;h<H;h++){potential_blocks[c][spin][g][h]=potential_rows[c][spin][g][h];for(int i=0;i<O;i++){
      potential_rows[c][spin][g][h][i]=potential[c][spin][g][h][i];for(int j=0;j<O;j++)potential[c][spin][g][h][i][j]=.2+.03*spin-.017*c+.011*g-.007*h+.013*i-.019*j;
    }}
  }}
  electronic.HNL=HV[0];electronic.iHNL=HV[1];
}
int main(int argc,char **argv){setup();CHECK(OpenMX_Solver_Run(callback,argc>1?argv[1]:NULL)==1);CHECK(comparisons==3888);printf("PASS: %d scalar nonlocal cases, four logical workers, three sizes, Gaussian strain and all complex spin/axis rows\n",comparisons);return 0;}

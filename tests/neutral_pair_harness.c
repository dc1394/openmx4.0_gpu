/* Independent polynomial moments and coordinate finite differences check
   production energy/stress two-center wrappers and their shared kernel.
   Radial interpolation is prescribed; production interpolation is exercised
   separately in full physical comparisons. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_atomic.h"
#include "openmx_core_neutral_grid.h"
#include "openmx_neutral_atom.h"
#define CHECK(c) do{if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(2);}}while(0)
static int WhatSpecies[4]={0,0,1,0},neighbors[4][3],images[4][3],*natn[4],*ncn[4];
static int TGN_EH0[2];
static double points[5][2][32];
static OpenMXNeutralGridTables grid_tables={&points[0][0][0],{5,2,32}};
static double dv_EH0[2]={.031,.043},coordinates[4][4],*Gxyz[4],distances[4][3],*Dis[4],atv[2][4];
static double charges[2]={1.5,2.7},radial[3][2][4];
static int mesh[2]={4,4},species_index[2]={1,0},value_calls,derivative_calls,pre_guard_logs;
static void near(double a,long double b){
  if(!isfinite(a)||fabsl((long double)a-b)>2e-11L*(1+fabsl(b))){fprintf(stderr,"values %.17g %.21Lg\n",a,b);CHECK(0);}
}
static int species(double charge,int count,double *logs,double *radii,double *potential){
  const int s=charge==charges[0]?0:1;CHECK(charge==charges[s]&&count==4);
  CHECK(logs==radial[0][s]&&radii==radial[1][s]&&potential==radial[2][s]);return s;
}
double VH_AtomF(double charge,int count,double logarithm,double r,double *logs,double *radii,double *potential){
  (void)species(charge,count,logs,radii,potential);CHECK(isfinite(logarithm)||(isinf(logarithm)&&logarithm<0));
  if(isinf(logarithm)){CHECK(r==1e-10);++pre_guard_logs;}
  ++value_calls;const double q=r*r;return charge*(potential[0]+potential[1]*q+potential[2]*q*q);
}
double Dr_VH_AtomF(double charge,int count,double logarithm,double r,double *logs,double *radii,double *potential){
  (void)species(charge,count,logs,radii,potential);(void)logarithm;++derivative_calls;
  return charge*(2*potential[1]*r+4*potential[2]*r*r*r);
}
#include "neutral_pair_native.inc"
/* Closed weighted polynomial moments: no production distance loop or
   derivative interpolation/reduction is reused by the oracle. */
static void moments(int base,int other,long double distance,long double *energy,long double *derivative){
  const int row=species_index[base];long double m0=0,mz=0,mq=0,mq2=0,mz2=0,mqz=0;
  for(int i=0;i<TGN_EH0[row];i++){
    const long double x=points[0][row][i],y=points[1][row][i],z=points[2][row][i],w=(long double)points[3][row][i]*points[4][row][i];
    const long double q=x*x+y*y+z*z;m0+=w;mz+=w*z;mq+=w*q;mq2+=w*q*q;mz2+=w*z*z;mqz+=w*q*z;
  }
  const long double a=radial[2][other][0],b=radial[2][other][1],c=radial[2][other][2],d=distance;
  const long double scale=(long double)charges[other]*dv_EH0[row];
  *energy=scale*(a*m0+b*(mq-2*d*mz+d*d*m0)+c*(mq2-4*d*mqz+2*d*d*mq+4*d*d*mz2-4*d*d*d*mz+d*d*d*d*m0));
  *derivative=scale*(b*(-2*mz+2*d*m0)+c*(-4*mqz+4*d*mq+8*d*mz2-12*d*d*mz+4*d*d*d*m0));
}
static void geometry(void){
  for(int a=1;a<=3;a++)for(int h=0;h<3;h++){
    const int b=neighbors[a][h],image=images[a][h];double q=0;
    for(int k=1;k<=3;k++){const double d=coordinates[a][k]-(coordinates[b][k]+atv[image][k]);q+=d*d;}
    distances[a][h]=sqrt(q);
  }
}
static int check_pairs(const OpenMXAtomicViews *atomic){
  int cases=0;
  for(int atom=1;atom<=3;atom++)for(int h=0;h<3;h++){
    const int other=neighbors[atom][h],base=WhatSpecies[atom],sp=WhatSpecies[other],row=species_index[base];
    const double distance=distances[atom][h];long double energy,derivative;moments(base,sp,distance,&energy,&derivative);
    for(int implementation=0;implementation<2;implementation++){
      double out[4]={99,98,97,96};value_calls=derivative_calls=0;
      if(implementation)StressNeutralPair(&grid_tables,atomic,atom,h,out);else EnergyNeutralPair(&grid_tables,atomic,atom,h,out);
      CHECK(value_calls==TGN_EH0[row]&&derivative_calls==(h?TGN_EH0[row]:0));near(out[0],energy);
      for(int d=0;d<3;d++){
        const double displacement=coordinates[atom][d+1]-(coordinates[other][d+1]+atv[images[atom][h]][d+1]);
        near(out[d+1],h?derivative*displacement/fmax(distance,1e-10):0);
        /* Independent coordinate gradient of the closed energy moment. */
        if(h&&distance>1e-6&&TGN_EH0[row]){
          const long double step=2e-6L;long double e[2],unused;
          for(int side=0;side<2;side++){
            long double q=0;for(int axis=0;axis<3;axis++){
              long double delta=(long double)coordinates[atom][axis+1]-((long double)coordinates[other][axis+1]+atv[images[atom][h]][axis+1]);
              if(axis==d)delta+=side?step:-step;q+=delta*delta;
            }
            moments(base,sp,sqrtl(q),e+side,&unused);
          }
          near(out[d+1],(e[1]-e[0])/(2*step));
        }
      }
      ++cases;
    }
  }return cases;
}
int main(void){
  OpenMXAtomicViews atomic={0};atomic.species.species=2;atomic.species.species_index=species_index;
  atomic.species.core_charge=charges;atomic.species.vps_mesh=mesh;
  atomic.vps_log_radii=(OpenMXRadialView2){&radial[0][0][0],{2,4}};
  atomic.vps_radii=(OpenMXRadialView2){&radial[1][0][0],{2,4}};
  atomic.hartree_potential=(OpenMXRadialView2){&radial[2][0][0],{2,4}};
  for(int s=0;s<2;s++){
    radial[2][s][0]=.7+.2*s;radial[2][s][1]=.13-.03*s;radial[2][s][2]=.017+.005*s;
  }
  for(int a=0;a<4;a++){natn[a]=neighbors[a];ncn[a]=images[a];Gxyz[a]=coordinates[a];Dis[a]=distances[a];}
  atv[1][1]=.23;atv[1][2]=-.11;atv[1][3]=.07;
  for(int a=1;a<=3;a++){
    coordinates[a][1]=.31*a;coordinates[a][2]=-.09*a*a;coordinates[a][3]=.17*a;
    for(int h=0;h<3;h++){neighbors[a][h]=h?(a+h-1)%3+1:a;images[a][h]=h==2;}
  }
  geometry();int cases=0;const int counts[]={0,1,7,23,3};
  for(int shape=0;shape<5;shape++){
    TGN_EH0[0]=counts[shape];TGN_EH0[1]=counts[4-shape];
    for(int s=0;s<2;s++)for(int i=0;i<TGN_EH0[s];i++){
      points[0][s][i]=.11*(1+i%3);points[1][s][i]=.05*(i%2)-.01;points[2][s][i]=-.4+.07*i;
      points[3][s][i]=.2+.013*i+.03*s;points[4][s][i]=.13*(i+1);
    }
    cases+=check_pairs(&atomic);
  }
  /* On-site r=0 reaches the interpolator with log=-inf and r clamped.
     Scalar small-distance semantics are legacy behavior, not a new model. */
  TGN_EH0[0]=TGN_EH0[1]=1;
  for(int s=0;s<2;s++){for(int d=0;d<3;d++)points[d][s][0]=0;points[3][s][0]=.4;points[4][s][0]=.7;}
  pre_guard_logs=0;cases+=check_pairs(&atomic);CHECK(pre_guard_logs>=6);
  /* Resolve both guards with a steep prescribed quadratic potential;
     neither a generic tolerance nor a finite difference can mask the clamp. */
  radial[2][0][1]=1e12;
  for(int d=1;d<=3;d++)coordinates[3][d]=coordinates[2][d];
  coordinates[3][1]+=3e-12;coordinates[3][3]+=4e-12;geometry();
  const double distance=distances[2][1],guarded=1e-10;
  CHECK(distance>0&&distance<1e-10);
  const long double scale=(long double)dv_EH0[species_index[1]]*.4L*.7L*charges[0];
  const long double r2=(long double)guarded*guarded;
  const long double guard_energy=scale*(radial[2][0][0]+radial[2][0][1]*r2+radial[2][0][2]*r2*r2);
  for(int implementation=0;implementation<2;implementation++){
    double out[4];if(implementation)StressNeutralPair(&grid_tables,&atomic,2,1,out);else EnergyNeutralPair(&grid_tables,&atomic,2,1,out);
    near(out[0],guard_energy);
    for(int d=0;d<3;d++){
      const double displacement=coordinates[2][d+1]-coordinates[3][d+1];
      near(out[d+1],scale*(2*radial[2][0][1]+4*radial[2][0][2]*r2)*distance*displacement/guarded);
    }++cases;
  }
  /* Cutoff has no guard. The same steep polynomial distinguishes its
     original small-radius value from the guarded pair value. */
  const OpenMXNeutralGridView tiny={1,dv_EH0[0],points[0][0],points[1][0],points[2][0],points[3][0],points[4][0]};
  OpenMXNeutralPotentialView potential;OpenMX_Neutral_PotentialBorrow(&atomic,0,&potential);
  double partial=0;OpenMX_Neutral_CutoffAccumulate(&tiny,&potential,distance,0,1,&partial);
  near(partial*dv_EH0[0],scale*(radial[2][0][0]+radial[2][0][1]*(long double)distance*distance+radial[2][0][2]*powl(distance,4)));
  /* Partition and accumulate the shared cutoff kernel over unequal and
     empty slices, retaining the caller's ordered reduction and initial values. */
  radial[2][0][1]=.13;TGN_EH0[0]=23;
  for(int i=0;i<23;i++){
    points[0][0][i]=.09*(1+i%3);points[1][0][i]=-.04*(i%2);points[2][0][i]=-.3+.045*i;
    points[3][0][i]=.3+.01*i;points[4][0][i]=.12*(i+1);
  }
  const OpenMXNeutralGridView grid={23,dv_EH0[0],points[0][0],points[1][0],points[2][0],points[3][0],points[4][0]};
  for(int nt=1;nt<=32;nt*=2){
    double sum=0;long double offset=0,energy,derivative;
    for(int t=0;t<nt;t++){
      double value=.003*(t+1);offset+=value;
      OpenMX_Neutral_CutoffAccumulate(&grid,&potential,.71,t*23/nt,(t+1)*23/nt,&value);sum+=value;
    }
    moments(1,0,.71L,&energy,&derivative);near(sum*grid.spacing,energy+offset*grid.spacing);
  }
  printf("PASS: %d production pair cases, both wrappers, permuted species, images, empty/grow/shrink grids, polynomial energy/gradient moments, finite differences log-before-guard, both small-distance guards and unguarded cutoff worker partitions\n",cases);
  return 0;
}

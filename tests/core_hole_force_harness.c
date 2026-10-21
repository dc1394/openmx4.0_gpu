#include "openmx_core_core_hole.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
static int SpinP_switch,Core_Hole_Atom=1,Core_Hole_J,Spe_MaxL_Basis[1]={4};
static char Core_Hole_Orbital[2];
static int basis[5]={2,2,2,2,1},*Spe_Num_Basis[1]={basis};
enum {capacity=64};
static double values[4][2][1][1][capacity],*rows[4][2][1][1],**blocks[4][2][1],***atoms[4][2],****overlap[4];
static int cases;
static void OpenMX_Solver_Failure(const char *error){fprintf(stderr,"%s\n",error);abort();}
/* OPENMX_CORE_HOLE_REFERENCE */
/* OPENMX_CORE_HOLE_PRODUCTION */
static int compare(OpenMXSolverInvocation *scope,void *unused,double *result)
{
  (void)unused;
  for(int missing=0;missing<2;missing++)for(int spin=0;spin<3;spin++)for(int angular=0;angular<4;angular++) {
    SpinP_switch=spin==2?3:spin;Core_Hole_Orbital[0]="spdf"[angular];
    for(int l=0;l<5;l++)basis[l]=(missing && l==angular)?0:(l==4?1:2);
    int count=0;for(int l=0;l<5;l++)count+=(2*l+1)*basis[l];
    for(Core_Hole_J=0;Core_Hole_J<=2*(2*angular+1);Core_Hole_J++)for(int sign=0;sign<2;sign++) {
      const double penalty=sign?-1.37:2.19;OpenMXCoreHoleView hole;char error[256];
      const OpenMXCoreHoleInput input={Core_Hole_Atom,SpinP_switch,1,angular,Core_Hole_J,4,count,basis,penalty};
      assert(openmx_core_core_hole_prepare(scope,&input,&hole,error,sizeof(error)));
      for(int reverse=0;reverse<2;reverse++)for(int direction=0;direction<3;direction++)for(int remote=0;remote<2;remote++) {
        const double factor=(double[]){1,-1,.5}[direction];const int kg=remote?7:Core_Hole_Atom;
        double expected[18],actual[18];for(int k=0;k<18;k++)expected[k]=actual[k]=sin(k*.37-.19);
        reference_dHCH_SO(expected,expected+1,expected+2,expected+3,expected+4,expected+5,
            expected+6,expected+7,expected+8,expected+9,expected+10,expected+11,
            expected+12,expected+13,expected+14,expected+15,expected+16,expected+17,
            factor,0,0,0,1,0,0,kg,0,reverse?-penalty:penalty,overlap);
        dHCH_SO(actual,actual+1,actual+2,actual+3,actual+4,actual+5,
            actual+6,actual+7,actual+8,actual+9,actual+10,actual+11,
            actual+12,actual+13,actual+14,actual+15,actual+16,actual+17,
            factor,0,0,0,1,0,0,kg,overlap,&hole,reverse);
        for(int k=0;k<18;k++)if(!isfinite(actual[k]) || fabs(actual[k]-expected[k])>3e-13*(1+fabs(expected[k]))) {
          fprintf(stderr,"core-hole l=%d J=%d spin=%d missing=%d reverse=%d sign=%g component=%d: %.17g != %.17g\n",
              angular,Core_Hole_J,SpinP_switch,missing,reverse,factor,k,actual[k],expected[k]);abort();
        }
        ++cases;
      }
    }
  }
  *result=19;return 1;
}
int main(void)
{
  for(int c=0;c<4;c++) {
    overlap[c]=atoms[c];
    for(int atom=0;atom<2;atom++) {
      atoms[c][atom]=blocks[c][atom];blocks[c][atom][0]=rows[c][atom][0];rows[c][atom][0][0]=values[c][atom][0][0];
      for(int k=0;k<capacity;k++)values[c][atom][0][0][k]=sin(.37*k-.29*c+.71*atom);
    }
  }
  double result;char error[256];
  assert(openmx_core_solver_run(1,compare,NULL,&result,error,sizeof(error)) && result==19);
  printf("PASS: %d legacy/core-hole spinor comparisons; s/p/d/f, all J states, spin 0/1/3, signed penalties, reversed derivatives, absent shells, remote atoms\n",cases);
}

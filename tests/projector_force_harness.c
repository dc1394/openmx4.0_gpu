#include "openmx_core_projector_force.h"
#include "openmx_core_electronic.h"
#include "openmx_host_simd.h"
#include "openmx_core_force_simd.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum {columns=64,slots=8,species_count=3,output_capacity=4};
static int SpinP_switch,wakg,Mc_AN,k,m,Mj_AN2,kl,n,kl2,Mj_AN,Mi_AN2,Matomnum=2,kl1,q_AN,ian,jan;
static int List_YOUSO[10]={0,0,0,0,0,0,0,output_capacity};
static int RMI1[4][3][3];
static int angular[species_count][slots+1],counts[species_count],components[species_count],relativistic[species_count];
static double energies[2][species_count][slots];
static OpenMXAtomicViews metadata;
static const OpenMXAtomicViews *atomic=&metadata;
static const OpenMXProjectorForceView *projectors;
static OpenMXElectronicViews matrices;
static const OpenMXElectronicViews *electronic=&matrices;
static size_t component_offsets[9],atom_offsets[33],block_offsets[97];
static int shapes[192];
static double values[2][4][4][3][2][columns];
static double *p1[2][4][4][3][2],**p2[2][4][4][3],***p3[2][4][4],****p4[2][4],*****p5[2];
static double ******DS_NL1=p5;
static dcomplex output[3][3][output_capacity][output_capacity];
static dcomplex *output_rows[3][3][output_capacity],**output_planes[3][3];
static dcomplex ***Hx=output_planes[0],***Hy=output_planes[1],***Hz=output_planes[2];
static dcomplex Complex(double r,double i){return (dcomplex){r,i};}
static void OpenMX_Solver_Failure(const char *error){fprintf(stderr,"%s\n",error);abort();}
/* Host SIMD policy seen by dHNL_block: the legacy Core block first, then the
   widest host SIMD mode, each against the frozen scalar reference. */
static int harness_host_simd_mode=OPENMX_HOST_SIMD_LEGACY;
int OpenMX_HostSimdMode(void){return harness_host_simd_mode;}
/* OPENMX_PROJECTOR_REFERENCE */
/* OPENMX_PROJECTOR_PRODUCTION */
static int comparisons;
static void reset(int scenario)
{
  wakg=scenario%species_count;Mc_AN=1+scenario%2;Mj_AN=1+(scenario/2)%2;Mj_AN2=scenario%2?3:2;
  Mi_AN2=scenario%3?1:3;k=scenario%3;kl=(scenario+1)%3;kl1=(scenario+2)%3;kl2=(scenario/2)%3;
  q_AN=scenario%3;ian=scenario==0?0:1+scenario%2;jan=scenario==1?0:1+(scenario/2)%2;
  for(int a=0;a<3;++a)for(int s=0;s<3;++s)for(int i=0;i<output_capacity;++i)for(int j=0;j<output_capacity;++j) {
    const int index=((a*3+s)*output_capacity+i)*output_capacity+j;
    output[a][s][i][j]=(dcomplex){sin(.27*index+.17*scenario),cos(.37*index-.21*scenario)};
  }
}
static int compare(OpenMXSolverInvocation *scope,void *unused,double *result)
{
  (void)unused;
  for(int layout=0;layout<8;++layout)for(int spin=0;spin<3;++spin)for(int so=0;so<2;++so)for(int energy=0;energy<3;++energy) {
    SpinP_switch=spin==2?3:spin;
    counts[0]=0;counts[1]=counts[2]=layout==0?0:layout<=4?1:layout==5?4:layout==6?6:8;
    for(int w=0;w<species_count;++w) {
      components[w]=0;
      for(int j=0;j<counts[w];++j) {
        const int l=layout<=4?layout-1:layout==5?j:(3*j+w)%4;
        angular[w][j+1]=l;components[w]+=2*l+1;
        for(int c=0;c<2;++c)energies[c][w][j]=energy==2?0:(energy?-1:1)*sin(.31*j+.67*w+.91*c+.13);
      }
    }
    OpenMXProjectorForceView views[2];char error[256];
    for(int r=0;r<2;++r) {
      for(int w=0;w<species_count;++w)relativistic[w]=r;
      assert(openmx_core_projector_force_prepare(scope,atomic,SpinP_switch,so,&views[r],error,sizeof(error)));
    }
    for(int scenario=0;scenario<12;++scenario)for(int which=0;which<9;++which) {
      const int r=relative_cases[which];projectors=&views[r];
      for(int w=0;w<species_count;++w)relativistic[w]=r;
      dcomplex expected[3][3][output_capacity][output_capacity];
      reset(scenario);legacy_block_call(which);memcpy(expected,output,sizeof(output));
      for(int flat=0;flat<2;++flat) {
      reset(scenario);production_block_call(which,flat);
      for(int a=0;a<3;++a)for(int s=0;s<3;++s)for(int i=0;i<output_capacity;++i)for(int j=0;j<output_capacity;++j) {
        const dcomplex x=output[a][s][i][j],y=expected[a][s][i][j];
        if(!isfinite(x.r) || !isfinite(x.i) || fabs(x.r-y.r)>3e-13*(1+fabs(y.r)) || fabs(x.i-y.i)>3e-13*(1+fabs(y.i))) {
          fprintf(stderr,"projector layout=%d spin=%d SO=%d energy=%d scenario=%d site=%d cell=%d/%d/%d/%d: (%g,%g) != (%g,%g)\n",
              layout,SpinP_switch,so,energy,scenario,which,a,s,i,j,x.r,x.i,y.r,y.i);abort();
        }
      }
      ++comparisons;
      }
    }
  }
  *result=19;return 1;
}
int main(void)
{
  metadata.species.species=species_count;metadata.species.radial_projectors=counts;
  metadata.species.projector_components=components;metadata.species.relativistic=relativistic;
  metadata.projector_angular=&angular[0][0];metadata.projector_angular_species=species_count;
  metadata.projector_angular_stride=slots+1;
  metadata.projector_energies=(OpenMXRadialView3){&energies[0][0][0],{2,species_count,slots}};
  for(int c=0;c<9;++c)component_offsets[c]=4*c;
  for(int a=0;a<33;++a)atom_offsets[a]=3*a;
  for(int b=0;b<97;++b)block_offsets[b]=2*columns*b;
  for(int b=0;b<96;++b){shapes[2*b]=2;shapes[2*b+1]=columns;}
  matrices.fields[OPENMX_ELECTRONIC_PROJECTORS]=(OpenMXElectronicMatrixView){2,4,8,32,96,8*4*3*2*columns,
      component_offsets,atom_offsets,block_offsets,shapes,&values[0][0][0][0][0][0]};
  for(int a=0;a<4;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)RMI1[a][i][j]=(a+i+2*j)%3;
  for(int a=0;a<3;++a)for(int s=0;s<3;++s) {
    output_planes[a][s]=output_rows[a][s];
    for(int i=0;i<output_capacity;++i)output_rows[a][s][i]=output[a][s][i];
  }
  for(int c=0;c<2;++c) {
    p5[c]=p4[c];
    for(int d=0;d<4;++d) {
      p4[c][d]=p3[c][d];
      for(int a=0;a<4;++a) {
        p3[c][d][a]=p2[c][d][a];
        for(int h=0;h<3;++h) {
          p2[c][d][a][h]=p1[c][d][a][h];
          for(int o=0;o<2;++o) {
            p1[c][d][a][h][o]=values[c][d][a][h][o];
            for(int j=0;j<columns;++j)values[c][d][a][h][o][j]=sin(.23*j-.41*c+.37*d+.29*a-.71*h+.53*o);
          }
        }
      }
    }
  }
  double result;char error[256];
  const int widest=openmx_core_host_simd_mode(NULL);
  for(int policy=0;policy<2;++policy) {
    harness_host_simd_mode=policy?widest:OPENMX_HOST_SIMD_LEGACY;
    assert(openmx_core_solver_run(1,compare,NULL,&result,error,sizeof(error)) && result==19);
  }
  printf("PASS: %d legacy/projector block comparisons; nine actual native blocks and row fallbacks, scalar/spinor, padded rectangular inputs/outputs, reversed terms, transposes, empty ranges/species, s/p/d/f, repeated channels, signed/zero energies; host SIMD policies legacy and %s\n",
      comparisons,openmx_core_host_simd_name(widest));
}

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_core_atomic.h"
static int mpi_comm_level1;
static int atomnum,Matomnum,SpeciesNum,*M2G,*WhatSpecies,*FNAN,**natn,Spe_Total_CNO[2];
#include "openmx_force_workspace.h"

enum { capacity=7 };
static int SpinP_switch,List_YOUSO[8],Hwan=0,Qwan=1,ian,jan;
static int Mh_AN=1,Mq_AN=2,kl=1,kl1=1,OMPID=1;
static double pref,dEx,dEy,dEz,dEx_threads[3],dEy_threads[3],dEz_threads[3];
static double *****CDM0,*****iDM0;
static OpenMXAtomicViews atomic_storage;
static const OpenMXAtomicViews *atomic=&atomic_storage;
static dcomplex data[9][3][capacity][capacity],*rows[9][3][capacity],**planes[9][3];
static dcomplex ***Hx0,***Hy0,***Hz0,***Hx1,***Hy1,***Hz1,***Hx,***Hy,***Hz;

/* OPENMX_FORCE_REFERENCE */
/* OPENMX_FORCE_PRODUCTION */

static void reset_force(void)
{
  dEx=.125;dEy=-.25;dEz=.5;
  for(int i=0;i<3;i++){dEx_threads[i]=-.375;dEy_threads[i]=.625;dEz_threads[i]=-.75;}
}
static void output(double force[12])
{
  force[0]=dEx;force[1]=dEy;force[2]=dEz;
  for(int i=0;i<3;i++){force[3+3*i]=dEx_threads[i];force[4+3*i]=dEy_threads[i];force[5+3*i]=dEz_threads[i];}
}
static int check_traces(const OpenMXForceWorkspace *work,void *unused,double *result)
{
  (void)unused;CDM0=work->density_rows;iDM0=work->imaginary_rows;
  for(size_t k=0;k<work->density.layout.elements;k++)work->density.values[k]=sin(.31*k+.73);
  for(size_t k=0;k<work->imaginary.layout.elements;k++)work->imaginary.values[k]=cos(.17*k-.41);
  for(int number=0;number<13;number++)for(int variant=0;variant<(reference_modes[number]==1?2:1);variant++) {
    SpinP_switch=variant?3:reference_modes[number];
    for(int weight=0;weight<4;weight++) {
      pref=reference_modes[number]==3?1.0:(double[]){.5,1,2,4}[weight];
      double expected[12],actual[12];reset_force();reference_trace(number);output(expected);
      reset_force();production_trace(number);output(actual);
      for(int c=0;c<12;c++)if(!isfinite(actual[c]) || fabs(actual[c]-expected[c])>2e-13*(1+fabs(expected[c]))) {
        fprintf(stderr,"trace %d, spin %d, %dx%d, weight %g, component %d: %.17g != %.17g\n",
            number,SpinP_switch,ian,jan,pref,c,actual[c],expected[c]);abort();
      }
    }
  }
  *result=17;return 1;
}
int main(void)
{
  List_YOUSO[7]=capacity;atomic_storage.species.contracted_orbitals=Spe_Total_CNO;
  for(int a=0;a<9;a++)for(int c=0;c<3;c++) {
    planes[a][c]=rows[a][c];
    for(int i=0;i<capacity;i++) {
      rows[a][c][i]=data[a][c][i];
      for(int j=0;j<capacity;j++)data[a][c][i][j]=(dcomplex){cos(a*.37-c*.13+i*.71-j*.29),sin(-a*.61+c*.23+i*.19+j*.43)};
    }
  }
  Hx0=planes[0];Hy0=planes[1];Hz0=planes[2];Hx1=planes[3];Hy1=planes[4];Hz1=planes[5];
  Hx=planes[6];Hy=planes[7];Hz=planes[8];
  int map[]={-1,1,2},species[]={-1,0,1},counts[]={-1,1,1};
  int first[]={1,2},second[]={2,1},*neighbors[]={NULL,first,second};
  const OpenMXForceInput input={2,2,0,2,3,1,map,species,Spe_Total_CNO,counts,neighbors};
  for(ian=1;ian<=5;ian++)for(jan=1;jan<=5;jan++) {
    Spe_Total_CNO[0]=ian;Spe_Total_CNO[1]=jan;double result;char error[256];
    if(!openmx_core_force_run(&input,1,check_traces,NULL,&result,error,sizeof(error))) {
      fprintf(stderr,"trace owner: %s\n",error);return 1;
    }
    assert(result==17);
  }
  puts("PASS: 13 native trace sites x 25 rectangular shapes; scalar/collinear/spinor, reverse pairs, padded derivatives, threaded accumulators");
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_core_local_basis.h"
static int SpinP_switch,Cnt_switch,List_YOUSO[60],mpi_comm_level1;
static int M2G[3]={0,1,2},F_G2M[4]={0,1,2,3},WhatSpecies[4]={0,0,1,0};
static int Spe_Total_NO[2]={2,3},Spe_Total_CNO[2],FNAN[4]={0,2,2,2};
static int n1[3]={1,2,3},n2[3]={2,3,1},n3[3]={3,1,2},*natn[4]={NULL,n1,n2,n3};
static int pair_indices[3][3][3],*pair_rows[3][3],**RMI1[3];
static int counts[2][3]={{2,1,1},{2,2,1}},*Spe_Specified_Num[2]={counts[0],counts[1]};
static int orbital_map[2][3][2]={{{0,1},{1,0},{0,0}},{{2,0},{1,2},{0,0}}};
static int *map_rows[2][3],**Spe_Trans_Orbital[2];
static double coefficient[4][3][3],flat_coefficient[4][3][3];
static double* reference_contraction_row(const OpenMXLocalBasisViews* v,int a,int o)
{ (void)v;return coefficient[a][o]; }
static void MPI_Abort(int c,int e){(void)c;(void)e;abort();}
static dcomplex Complex(double re,double im){return (dcomplex){re,im};}

typedef struct {
  double data[4][4][3][3][3];
  double *row[4][4][3][3],**block[4][4][3],***atom[4][4],****component[4];
} Overlap;
typedef struct { double data[3][3][3],*row[3][3],**component[3]; } RealOutput;
typedef struct { dcomplex data[2][2][3][3],*row[2][2][3],**block[2][2],***component[2]; } ComplexOutput;
static void real_output(RealOutput* o) {
  for(int s=0;s<3;s++) {o->component[s]=o->row[s];for(int i=0;i<3;i++) {
    o->row[s][i]=o->data[s][i];for(int j=0;j<3;j++)o->data[s][i][j]=.3+s*.1+i*.01-j*.02;
  }}
}
static void complex_output(ComplexOutput* o) {
  for(int s=0;s<2;s++){o->component[s]=o->block[s];for(int t=0;t<2;t++) {
    o->block[s][t]=o->row[s][t];for(int i=0;i<3;i++){
      o->row[s][t][i]=o->data[s][t][i];for(int j=0;j<3;j++)o->data[s][t][i][j]=(dcomplex){.2+s*.1-i*.02,.1+t*.2+j*.01};
    }
  }}
}
static void equal(double a,double b) {
  if(!isfinite(a) || !isfinite(b) || fabs(a-b)>2e-13*(1+fabs(b))){fprintf(stderr,"Hubbard force: %.17g != %.17g\n",a,b);abort();}
}
static void compare_real(RealOutput* a,RealOutput* b) {for(int s=0;s<3;s++)for(int i=0;i<3;i++)for(int j=0;j<3;j++)equal(a->data[s][i][j],b->data[s][i][j]);}
static void compare_complex(ComplexOutput* a,ComplexOutput* b) {for(int s=0;s<2;s++)for(int t=0;t<2;t++)for(int i=0;i<3;i++)for(int j=0;j<3;j++){equal(a->data[s][t][i][j].r,b->data[s][t][i][j].r);equal(a->data[s][t][i][j].i,b->data[s][t][i][j].i);}}

/* Reference routines are inserted here, followed by the shared production header. */

/* OPENMX_HUBBARD_REFERENCE */
#include "openmx_hubbard_force.h"
#include "openmx_hubbard_dual_force.h"

static double test_value(int index) { return sin(.17*index)+.09*cos(.37*index); }

static void check_dual(void) {
  enum {rows=11,columns=17,extent=rows*rows+columns*columns};
  size_t offsets[4]={0,0,rows*rows,extent};
  int shapes[6]={0,0,rows,rows,columns,columns};
  double real_potential[2*extent];
  dcomplex complex_potential[4*extent];
  for(int i=0;i<2*extent;++i)real_potential[i]=test_value(i+5);
  for(int i=0;i<4*extent;++i)complex_potential[i]=(dcomplex){test_value(i+1),test_value(i+37)};
  OpenMXLocalBasisViews basis={0};
  basis.real[OPENMX_LOCAL_POTENTIAL]=(OpenMXLocalRealMatrix){{2,3,extent,2*extent,offsets,shapes},real_potential};
  basis.complex_values[OPENMX_LOCAL_COMPLEX_POTENTIAL]=(OpenMXLocalComplexMatrix){{4,3,extent,4*extent,offsets,shapes},complex_potential};
  double derivative[3][rows][columns],result[3][rows][columns];
  double *derivative_rows[3][rows], *result_rows[3][rows], **output[3];
  OpenMXHubbardDualMatrices block={0};
  for(int axis=0;axis<3;++axis) {
    block.derivative[axis]=derivative_rows[axis];output[axis]=result_rows[axis];
    for(int i=0;i<rows;++i) {
      derivative_rows[axis][i]=derivative[axis][i];result_rows[axis][i]=result[axis][i];
      for(int j=0;j<columns;++j) {derivative[axis][i][j]=test_value(axis*rows*columns+i*columns+j);result[axis][i][j]=NAN;}
    }
  }
  for(int spin=0;spin<2;++spin) {
    OpenMX_HubbardDualReal(&basis,spin,1,2,rows,columns,block.derivative,output);
    for(int axis=0;axis<3;++axis)for(int i=0;i<rows;++i)for(int j=0;j<columns;++j) {
      double scalar=0.0;
      for(int k=0;k<rows;++k)scalar+=openmx_local_potential_row(&basis,spin,1,i)[k]*derivative[axis][k][j];
      for(int k=0;k<columns;++k)scalar+=openmx_local_potential_row(&basis,spin,2,k)[j]*derivative[axis][i][k];
      equal(result[axis][i][j],scalar);
    }
  }

  double density[2][6][rows*columns],*density_rows[2][6][columns];
  for(int direction=0;direction<2;++direction)for(int c=0;c<6;++c) {
    const int n=direction?columns:rows, width=direction?rows:columns;
    for(int i=0;i<n;++i)density_rows[direction][c][i]=density[direction][c]+i*width;
    for(int i=0;i<n*width;++i)density[direction][c][i]=test_value(direction*1000+c*7+i);
    if(c<4)block.density[direction][c]=density_rows[direction][c];
    else block.imaginary_density[direction][c-4]=density_rows[direction][c];
  }
  double actual[3]={.2,-.08,.9},expected[3]={.2,-.08,.9},*force[3]={actual,actual+1,actual+2};
  const int spin_rows[3]={0,1,0},spin_cols[3]={0,1,1};
  for(int i=0;i<rows;++i)for(int j=0;j<columns;++j)for(int direction=0;direction<2;++direction) {
    const int row=direction?j:i,column=direction?i:j;
    const size_t index=(size_t)row*(direction?rows:columns)+column;
    for(int axis=0;axis<3;++axis) {
      double real[3]={0},imag[3]={0};
      for(int c=0;c<3;++c) {
        const int s=spin_rows[c],t=spin_cols[c];
        for(int term=0;term<2;++term) {
          const int right=term!=direction;
          for(int k=0;k<(right?columns:rows);++k) {
            const int urow=right?(direction?j:k):(direction?k:i);
            const int ucol=right?(direction?k:j):(direction?i:k);
            const dcomplex v=openmx_local_complex_potential_row(&basis,s,t,right?2:1,urow)[ucol];
            const double d=right?derivative[axis][i][k]:derivative[axis][k][j];
            real[c]+=v.r*d;imag[c]+=v.i*d;
          }
        }
      }
      const double (*dm)[rows*columns]=density[direction];
      const double d=real[0]*dm[0][index]+real[1]*dm[1][index]+2.0*real[2]*dm[2][index]
        -imag[0]*dm[4][index]-imag[1]*dm[5][index]-2.0*imag[2]*dm[3][index];
      expected[axis]+=.5*d;
    }
  }
  OpenMX_HubbardDualNoncollinear(&basis,1,2,rows,columns,&block,force);
  OpenMX_HubbardDualNoncollinear(&basis,1,2,0,columns,NULL,force);
  for(int axis=0;axis<3;++axis)equal(actual[axis],expected[axis]);
}

int main(void) {
  Overlap overlap;
  double potential[2][4][3][3],*potential_rows[2][4][3],**potential_atoms[2][4],***v_eff[2];
  dcomplex complex_potential[2][2][4][3][3],*complex_rows[2][2][4][3],**complex_atoms[2][2][4],***complex_spins[2][2],****NC_v_eff[2];
  double real_values[2*18];dcomplex complex_values[4*18];
  size_t offsets[5]={0,1,5,14,18};int shapes[8]={1,1,2,2,3,3,2,2};
  OpenMXLocalBasisViews local={0};
  local.real[OPENMX_LOCAL_POTENTIAL]=(OpenMXLocalRealMatrix){{2,4,18,36,offsets,shapes},real_values};
  local.complex_values[OPENMX_LOCAL_COMPLEX_POTENTIAL]=(OpenMXLocalComplexMatrix){{4,4,18,72,offsets,shapes},complex_values};
  local.tensors[OPENMX_LOCAL_CONTRACTION]=(OpenMXLocalTensor3){4,3,3,&flat_coefficient[0][0][0]};
  List_YOUSO[7]=3;
  for(int a=0;a<4;a++)for(int i=0;i<3;i++)for(int p=0;p<3;p++)coefficient[a][i][p]=flat_coefficient[a][i][p]=.5+a*.11-i*.17+p*.13;
  for(int w=0;w<2;w++){Spe_Trans_Orbital[w]=map_rows[w];for(int i=0;i<3;i++)map_rows[w][i]=orbital_map[w][i];}
  for(int s=0;s<4;s++){overlap.component[s]=overlap.atom[s];for(int a=0;a<4;a++) {
    overlap.atom[s][a]=overlap.block[s][a];for(int h=0;h<3;h++) {
      overlap.block[s][a][h]=overlap.row[s][a][h];for(int i=0;i<3;i++) {
        overlap.row[s][a][h][i]=overlap.data[s][a][h][i];
        for(int j=0;j<3;j++)overlap.data[s][a][h][i][j]=sin(s*.3+a*.1-h*.2+i*.31+j*.11);
      }
    }
  }}
  for(int s=0;s<2;s++){v_eff[s]=potential_atoms[s];NC_v_eff[s]=complex_spins[s];for(int t=0;t<2;t++){
    complex_spins[s][t]=complex_atoms[s][t];for(int a=0;a<4;a++){
      potential_atoms[s][a]=potential_rows[s][a];complex_atoms[s][t][a]=complex_rows[s][t][a];
      const int n=a?Spe_Total_NO[WhatSpecies[a]]:1;
      for(int i=0;i<3;i++){
        potential_rows[s][a][i]=potential[s][a][i];complex_rows[s][t][a][i]=complex_potential[s][t][a][i];
        for(int j=0;j<3;j++) {
          double v=.1+s*.4+a*.3+i*.15-j*.07;potential[s][a][i][j]=v;
          complex_potential[s][t][a][i][j]=(dcomplex){v+t*.17,.1+s*.2-t*.1+a*.03+i*.02-j*.01};
          if(i<n && j<n){openmx_local_potential_row(&local,s,a,i)[j]=v;openmx_local_complex_potential_row(&local,s,t,a,i)[j]=complex_potential[s][t][a][i][j];}
        }
      }
    }
  }}
  double transpose[9*3*3];
  OpenMXLocalTensor3 scratch={9,3,3,transpose};
  int cases=0;
  for(int contraction=0;contraction<2;contraction++)for(int mode=0;mode<3;mode++)for(int missing=0;missing<2;missing++) {
    Cnt_switch=contraction;SpinP_switch=mode==2?3:mode;
    for(int w=0;w<2;w++)Spe_Total_CNO[w]=contraction?w+1:Spe_Total_NO[w];
    for(int a=1;a<=2;a++){RMI1[a]=pair_rows[a];for(int h=0;h<3;h++){
      pair_rows[a][h]=pair_indices[a][h];for(int q=0;q<3;q++){
        int other=natn[a][h],target=natn[a][q],index=0;while(natn[other][index]!=target)index++;
        pair_indices[a][h][q]=(missing && h==1 && q==2)?-1:index;
      }
    }}
    for(int a=1;a<=2;a++)for(int h=0;h<3;h++)for(int q=0;q<3;q++) {
      if(mode<2) {
        RealOutput actual[3],expected[3];for(int i=0;i<3;i++){real_output(actual+i);real_output(expected+i);}
        reference_dH_U_full(a,h,q,overlap.component,v_eff,expected[0].component,expected[1].component,expected[2].component,&local);
        /* Reused buffers contain unrelated data: no entry may rely on calloc. */
        for(size_t k=0;k<sizeof(transpose)/sizeof(*transpose);k++)transpose[k]=NAN;
        dH_U_full(a,h,q,overlap.component,actual[0].component,actual[1].component,actual[2].component,&local,scratch);
        for(int i=0;i<3;i++)compare_real(actual+i,expected+i);
      }else {
        ComplexOutput actual[3],expected[3];for(int i=0;i<3;i++){complex_output(actual+i);complex_output(expected+i);}
        reference_dH_U_NC_full(a,h,q,overlap.component,NC_v_eff,expected[0].component,expected[1].component,expected[2].component);
        dH_U_NC_full(a,h,q,overlap.component,actual[0].component,actual[1].component,actual[2].component,&local);
        for(int i=0;i<3;i++)compare_complex(actual+i,expected+i);
      }
      cases++;
    }
  }
  check_dual();
  printf("PASS: %d Hubbard derivative cases and both dual-projector orientations; mixed species, contraction, spin 0/1/3 and halo blocks\n",cases);
}

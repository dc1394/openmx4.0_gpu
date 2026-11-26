#ifndef OPENMX_HUBBARD_DUAL_FORCE_H
#define OPENMX_HUBBARD_DUAL_FORCE_H

#include "openmx_core_local_basis.h"

/* U_a dS + dS U_b for the collinear dual-projector force. Each SIMD lane
   computes a different output column with its original summation order. */
static void OpenMX_HubbardDualReal(const OpenMXLocalBasisViews *basis,int spin,int atom,int neighbor,
    int rows,int columns,double **const derivative[3],double **const output[3])
{
    if (!rows || !columns) return;
    const size_t stride=(size_t)basis->real[OPENMX_LOCAL_POTENTIAL].layout.shapes[2*neighbor+1];
    const double *right=openmx_local_potential_row(basis,spin,neighbor,0);
    for (int i=0;i<rows;++i) {
        const double *left=openmx_local_potential_row(basis,spin,atom,i);
        for (int axis=0;axis<3;++axis) {
            double *restrict result=output[axis][i];
            double **dS=derivative[axis];
            for (int j=0;j<columns;++j) result[j]=0.0;
            for (int k=0;k<rows;++k) {
                const double u=left[k], *d=dS[k];
#pragma omp simd
                for (int j=0;j<columns;++j) result[j]+=u*d[j];
            }
            for (int k=0;k<columns;++k) {
                const double d=dS[i][k], *u=right+(size_t)k*stride;
#pragma omp simd
                for (int j=0;j<columns;++j) result[j]+=u[j]*d;
            }
        }
    }
}

typedef struct {
    double **derivative[3];
    double **density[2][4];
    double **imaginary_density[2][2];
} OpenMXHubbardDualMatrices;

static inline void OpenMX_HubbardDualLeft(const dcomplex *u,size_t stride,int row,int rows,
    double *const *dS,int first,int count,int transposed,double *restrict re,double *restrict im)
{
    const dcomplex *urow=u+(size_t)row*stride;
    for (int k=0;k<rows;++k) {
        const dcomplex c=transposed ? u[(size_t)k*stride+row] : urow[k];
        const double *d=dS[k]+first;
#pragma omp simd
        for (int j=0;j<count;++j) {
            re[j]+=c.r*d[j];
            im[j]+=c.i*d[j];
        }
    }
}

static inline void OpenMX_HubbardDualRight(const dcomplex *u,size_t stride,int columns,
    const double *dS,int first,int count,int transposed,double *restrict re,double *restrict im)
{
    for (int k=0;k<columns;++k) {
        const double d=dS[k];
        const dcomplex *urow=u+(size_t)k*stride+first;
#pragma omp simd
        for (int j=0;j<count;++j) {
            const dcomplex c=transposed ? u[(size_t)(first+j)*stride+k] : urow[j];
            re[j]+=c.r*d;
            im[j]+=c.i*d;
        }
    }
}

/* Retain both orientations until force accumulation so every component
   keeps the legacy pair, row, column and direction order. */
static void OpenMX_HubbardDualNoncollinear(const OpenMXLocalBasisViews *basis,int atom,int neighbor,
    int rows,int columns,const OpenMXHubbardDualMatrices *block,double *const force[3])
{
    if (!rows || !columns) return;
    enum { columns_per_tile=8 };
    const OpenMXLocalComplexMatrix *potential=&basis->complex_values[OPENMX_LOCAL_COMPLEX_POTENTIAL];
    const size_t left_stride=(size_t)potential->layout.shapes[2*atom+1];
    const size_t right_stride=(size_t)potential->layout.shapes[2*neighbor+1];
    const dcomplex *left[3]={openmx_local_complex_potential_row(basis,0,0,atom,0),
                             openmx_local_complex_potential_row(basis,1,1,atom,0),
                             openmx_local_complex_potential_row(basis,0,1,atom,0)};
    const dcomplex *right[3]={openmx_local_complex_potential_row(basis,0,0,neighbor,0),
                              openmx_local_complex_potential_row(basis,1,1,neighbor,0),
                              openmx_local_complex_potential_row(basis,0,1,neighbor,0)};
    for (int i=0;i<rows;++i) for (int first=0;first<columns;first+=columns_per_tile) {
        const int count=columns-first<columns_per_tile ? columns-first : columns_per_tile;
        double re[2][3][3][columns_per_tile]={0}, im[2][3][3][columns_per_tile]={0};
        for (int axis=0;axis<3;++axis) for (int c=0;c<3;++c) {
            double **dS=block->derivative[axis];
            OpenMX_HubbardDualLeft(left[c],left_stride,i,rows,dS,first,count,0,re[0][axis][c],im[0][axis][c]);
            OpenMX_HubbardDualRight(right[c],right_stride,columns,dS[i],first,count,0,re[0][axis][c],im[0][axis][c]);
            OpenMX_HubbardDualRight(right[c],right_stride,columns,dS[i],first,count,1,re[1][axis][c],im[1][axis][c]);
            OpenMX_HubbardDualLeft(left[c],left_stride,i,rows,dS,first,count,1,re[1][axis][c],im[1][axis][c]);
        }
        for (int j=0;j<count;++j) for (int direction=0;direction<2;++direction) {
            const int row=direction ? first+j : i, column=direction ? i : first+j;
            const double rho00=block->density[direction][0][row][column];
            const double rho11=block->density[direction][1][row][column];
            const double rho01=block->density[direction][2][row][column];
            const double sigma01=block->density[direction][3][row][column];
            const double sigma00=block->imaginary_density[direction][0][row][column];
            const double sigma11=block->imaginary_density[direction][1][row][column];
            for (int axis=0;axis<3;++axis) {
                const double d=re[direction][axis][0][j]*rho00 + re[direction][axis][1][j]*rho11
                    + 2.0*re[direction][axis][2][j]*rho01 - im[direction][axis][0][j]*sigma00
                    - im[direction][axis][1][j]*sigma11 - 2.0*im[direction][axis][2][j]*sigma01;
                *force[axis]+=0.5*d;
            }
        }
    }
}

#endif

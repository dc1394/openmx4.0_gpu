#!/usr/bin/env python3
"""Exercise native NC-cluster CWF bra/adjoint algebra without MPI or a GPU.

The overlap stores and complete projected-Hamiltonian block are extracted
from production. A small dense GEMM adapter interprets the actual transpose
flags. S-orthonormal spinors with independent complex phases test the CWF
Hamiltonian, spectrum and reconstructed guiding orbitals. Unit-unitary
projection has an exactly known polar factor, requiring no LAPACK dependency.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command

ROOT = Path(__file__).resolve().parents[1]


def blocks(source):
    start = source.index('    /* store <Bloch functions|basis functions> */')
    end = source.index('  } // i', start)
    bra = source[start:end]
    start = source.index('  /* E x Hs -> Cs */')
    end = source.index('  for (i=0;', source.index('  /* Hs^dag x Cs -> Ws */', start))
    return bra, source[start:end]


PREFIX = r'''
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_complex.h"
#include "openmx_core_cwf.h"
#define CHECK(c) do { if(!(c)) { fprintf(stderr,"FAIL: line %d: %s\n",__LINE__,#c); exit(2); } } while(0)
#define F77_NAME(lower,upper) lower
static void Cblacs_barrier(int context,const char *scope) { (void)context;(void)scope; }
static double complex value(const dcomplex *a,int row,int col,int ld,char op) {
  const dcomplex z=op=='N'?a[col*ld+row]:a[row*ld+col];
  const double complex v=z.r+I*z.i; return op=='C'?conj(v):v;
}
static void pzgemm(const char *ta,const char *tb,const int *m,const int *n,const int *k,
    const dcomplex *alpha,const dcomplex *a,const int *ia,const int *ja,const int *da,
    const dcomplex *b,const int *ib,const int *jb,const int *db,const dcomplex *beta,
    dcomplex *c,const int *ic,const int *jc,const int *dc) {
  CHECK(*ia==1 && *ja==1 && *ib==1 && *jb==1 && *ic==1 && *jc==1);
  CHECK((*ta=='N'||*ta=='T'||*ta=='C') && (*tb=='N'||*tb=='T'||*tb=='C'));
  for(int j=0;j<*n;j++)for(int i=0;i<*m;i++) {
    double complex z=0;
    for(int h=0;h<*k;h++)z+=value(a,i,h,da[8],*ta)*value(b,h,j,db[8],*tb);
    z=(alpha->r+I*alpha->i)*z+(beta->r+I*beta->i)*value(c,i,j,dc[8],'N');
    c[j*dc[8]+i]=(dcomplex){creal(z),cimag(z)};
  }
}
static void bra(double complex up,double complex down,double complex *InProd_BasisFunc) {
  int m=0,n2=2,MP[2]={0,1},Gc_AN=1,i=0,p;
  double complex csum1=up,csum2=down;
'''
HSTART = r'''
}
static void projected(dcomplex *Hs,const double *ko,dcomplex *Ws) {
  int i,j,ig,ONE=1,MaxN=2,TNum_CWFs=2,na_rows_CWF3=2,na_cols_CWF3=2;
  int np_rows_CWF3=1,nblk_CWF3=1,my_prow_CWF3=0,ictxt1_CWF1=0;
  int desc_CWF3[9]={0,0,0,0,0,0,0,0,2},desc_CWF1[9]={0,0,0,0,0,0,0,0,2};
  dcomplex Cs[4]={{0}},alpha={1,0},beta={0,0};
  OpenMXCWFGrids metadata={0};OpenMXCWFGrids *grids=&metadata;
  OpenMXCWFGrid *projection=&grids->matrix[OPENMX_CWF_PROJECTION_GRID];
  OpenMXCWFGrid *cwf_projection=projection;
  OpenMXCWFGrid *cwf_hopping=&grids->matrix[OPENMX_CWF_HOPPING_GRID];
  projection->local_rows=2;projection->local_columns=2;
  projection->process_rows=1;projection->block_size=1;
  projection->descriptor[8]=2;
  grids->matrix[OPENMX_CWF_HOPPING_GRID].descriptor[8]=2;

'''
DRIVER = r'''
}
static void close_complex(double complex a,double complex b) { CHECK(cabs(a-b)<2e-12); }
static double complex zvalue(dcomplex z) { return z.r+I*z.i; }
int main(void) {
  /* S=diag(2,1/2); columns of psi and guide are S-orthonormal. */
  const double root=sqrt(0.5),c=cos(0.43),s=sin(0.43);
  const double complex U[4]={c,I*s,I*s,c}; /* column-major unitary */
  const double complex V[4]={root,root*I,root*I,root};
  const double energy[3]={0,-3.25,1.125},metric[2]={2,0.5};
  double complex guide[4],expected_h[4]={0};
  dcomplex reference[4]={{0}};
  for(int j=0;j<2;j++)for(int i=0;i<2;i++) {
    guide[j*2+i]=0;
    for(int k=0;k<2;k++)guide[j*2+i]+=U[k*2+i]/sqrt(metric[i])*V[j*2+k];
    for(int k=0;k<2;k++)expected_h[j*2+i]+=conj(V[i*2+k])*energy[k+1]*V[j*2+k];
  }
  for(int trial=0;trial<65;trial++) {
    double complex psi[4],projection[4];dcomplex q[4],h[4]={{0}};
    for(int j=0;j<2;j++) {
      const double complex phase=cexp(I*((j?0.31:-0.57)*trial+0.19));
      for(int i=0;i<2;i++)psi[j*2+i]=U[j*2+i]/sqrt(metric[i])*phase;
      double complex a[2];bra(metric[0]*psi[j*2],metric[1]*psi[j*2+1],a);
      for(int k=0;k<2;k++) {
        projection[k*2+j]=a[0]*guide[k*2]+a[1]*guide[k*2+1];
        q[k*2+j]=(dcomplex){creal(projection[k*2+j]),cimag(projection[k*2+j])};
      }
    }
    /* With a unitary projection the polar factor is the projection itself. */
    for(int j=0;j<2;j++)for(int i=0;i<2;i++) {
      double complex gram=0,w=0;
      for(int k=0;k<2;k++) {
        gram+=conj(projection[i*2+k])*projection[j*2+k];
        w+=psi[k*2+i]*projection[j*2+k];
      }
      close_complex(gram,i==j?1:0);close_complex(w,guide[j*2+i]);
    }
    projected(q,energy,h);
    for(int j=0;j<2;j++)for(int i=0;i<2;i++) {
      close_complex(zvalue(h[j*2+i]),expected_h[j*2+i]);
      close_complex(zvalue(h[j*2+i]),conj(zvalue(h[i*2+j])));
      if(trial)close_complex(zvalue(h[j*2+i]),zvalue(reference[j*2+i]));
    }
    close_complex(zvalue(h[0])+zvalue(h[3]),energy[1]+energy[2]);
    close_complex(zvalue(h[0])*zvalue(h[3])-zvalue(h[1])*zvalue(h[2]),energy[1]*energy[2]);
    for(int i=0;i<4;i++)reference[i]=h[i];
  }
  puts("PASS: native CWF bra/adjoint, 65 independent phases, Hermiticity, spectrum and S-metric reconstruction");
  return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negative-control', action='store_true')
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'source/Cluster_DFT_NonCol_CWF.c')
    args = parser.parse_args()
    bra, hamiltonian = blocks(args.source.read_text())
    fixtures = [('native', bra, hamiltonian, True)]
    if args.negative_control:
        old_bra = bra.replace('conj(csum1)', 'csum1').replace('conj(csum2)', 'csum2')
        old_h = hamiltonian.replace('"C","N"', '"T","N"')
        assert old_bra != bra and old_h != hamiltonian
        fixtures += [('ket-overlap', old_bra, hamiltonian, False),
                     ('ordinary-transpose', bra, old_h, False)]
    with tempfile.TemporaryDirectory(prefix='openmx-cwf-projection-') as directory:
        folder = Path(directory)
        for label, b, h, success in fixtures:
            source, exe = folder / (label+'.c'), folder / label
            source.write_text(PREFIX+b+HSTART+h+DRIVER)
            subprocess.run(compiler_command('c')+['-I'+str(ROOT/'core/include'),
                           '-x','c',str(source),'-lm','-o',str(exe)],check=True)
            result = subprocess.run([str(exe)],capture_output=True,text=True,
                                    preexec_fn=bounded_stack,timeout=60)
            if success:
                print(result.stdout,end='');print(result.stderr,end='')
                assert result.returncode==0,(label,result.returncode)
            else:
                assert result.returncode==2 and 'FAIL:' in result.stderr,(label,result)
                print('PASS: rejected regression',label)


if __name__=='__main__':
    main()

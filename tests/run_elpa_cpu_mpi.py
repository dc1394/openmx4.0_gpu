#!/usr/bin/env python3
"""Check the production CPU ELPA adapter against LAPACK with real MPI.

Uses the selected native build's compiler and linked ELPA release. No GPU is
used. Run after building OpenMX, for example:
    python3 tests/run_elpa_cpu_mpi.py --native-source build/cuda/native/source
The rectangular process grids, disjoint subcommunicators, partial vectors and
padded, unequal matrix leading dimensions exercise the legacy caller ABI.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_elpa_cpu.h"

extern void dsyev_(char *, char *, int *, double *, int *, double *, double *, int *, int *);
extern void zheev_(char *, char *, int *, double complex *, int *, double *, double complex *, int *, double *, int *);

static int world_rank, checked;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"rank %d line %d: %s\n",world_rank,__LINE__,#x); MPI_Abort(MPI_COMM_WORLD,1); } } while (0)

static int local_count(int n, int block, int coord, int procs)
{
    int count=0;
    for (int i=0;i<n;i++) count+=(i/block)%procs==coord;
    return count;
}

static int global_index(int local, int block, int coord, int procs)
{
    return (local/block*procs+coord)*block+local%block;
}

static double complex entry(int i, int j, int is_complex)
{
    double real=cos((double)(i+j+2))*.03+sin((double)((i+1)*(j+1)))*.02;
    if(i==j) real+=2.0+.3*i;
    return real + (is_complex ? I*.025*sin((double)(i-j)) : 0.0);
}

static void run_case(MPI_Comm parent, int np_rows, int np_cols,
                     int solver, int is_complex, int n, int nev, int block)
{
    MPI_Comm row_comm,col_comm;
    int parent_rank,parent_size;
    MPI_Comm_rank(parent,&parent_rank); MPI_Comm_size(parent,&parent_size);
    CHECK(parent_size==np_rows*np_cols);
    int prow=parent_rank%np_rows,pcol=parent_rank/np_rows;
    MPI_Comm_split(parent,pcol,prow,&row_comm);
    MPI_Comm_split(parent,prow,pcol,&col_comm);
    int fr=MPI_Comm_c2f(row_comm),fc=MPI_Comm_c2f(col_comm),fp=MPI_Comm_c2f(parent);
    int rows=local_count(n,block,prow,np_rows),cols=local_count(n,block,pcol,np_cols);
    /* Different ranks may independently need repacking. All must still
       participate in the same allocation and ELPA collectives. */
    int lda=rows+2,ldq=(n==11 || (n==23 && pcol%2==0))?lda:rows+5;
    double *ev=calloc(n,sizeof(*ev)),*reference=calloc(n,sizeof(*reference));
    double complex *full=calloc((size_t)n*n,sizeof(*full));
    double complex *vectors=calloc((size_t)n*n,sizeof(*vectors));
    CHECK(ev && reference && full && vectors);
    for(int j=0;j<n;j++)for(int i=0;i<n;i++)full[(size_t)j*n+i]=entry(i,j,is_complex);
    char job='N',uplo='U'; int info,lwork=8*n*n+100;
    if(is_complex) {
        double complex *work=calloc(lwork,sizeof(*work));
        double *rwork=calloc(3*n+1,sizeof(*rwork)); CHECK(work && rwork);
        zheev_(&job,&uplo,&n,full,&n,reference,work,&lwork,rwork,&info);
        free(work);free(rwork);
    } else {
        double *real=calloc((size_t)n*n,sizeof(*real)),*work=calloc(lwork,sizeof(*work)); CHECK(real && work);
        for(int i=0;i<n*n;i++)real[i]=creal(full[i]);
        dsyev_(&job,&uplo,&n,real,&n,reference,work,&lwork,&info);
        free(real);free(work);
    }
    CHECK(info==0);
    const double sentinel=-987654.25;
    if(is_complex) {
        double complex *a=malloc((size_t)lda*(cols?cols:1)*sizeof(*a));
        double complex *q=malloc((size_t)ldq*(cols?cols:1)*sizeof(*q)); CHECK(a && q);
        for(int k=0;k<lda*(cols?cols:1);k++)a[k]=sentinel;
        for(int k=0;k<ldq*(cols?cols:1);k++)q[k]=sentinel;
        for(int j=0;j<cols;j++)for(int i=0;i<rows;i++)a[(size_t)j*lda+i]=entry(global_index(i,block,prow,np_rows),global_index(j,block,pcol,np_cols),1);
        if(solver==1) solve_evp_complex_(&n,&nev,a,&lda,ev,q,&ldq,&block,&fr,&fc);
        else elpa_solve_evp_complex_2stage_double_impl_(&n,&nev,a,&lda,ev,q,&ldq,&block,&cols,&fr,&fc,&fp);
        for(int j=0;j<cols;j++) {
            for(int i=rows;i<ldq;i++)CHECK(q[(size_t)j*ldq+i]==sentinel);
            for(int i=rows;i<lda;i++)CHECK(a[(size_t)j*lda+i]==sentinel);
            int gj=global_index(j,block,pcol,np_cols);
            if(gj<nev)for(int i=0;i<rows;i++)vectors[(size_t)gj*n+global_index(i,block,prow,np_rows)]=q[(size_t)j*ldq+i];
        }
        free(a);free(q);
    } else {
        double *a=malloc((size_t)lda*(cols?cols:1)*sizeof(*a));
        double *q=malloc((size_t)ldq*(cols?cols:1)*sizeof(*q)); CHECK(a && q);
        for(int k=0;k<lda*(cols?cols:1);k++)a[k]=sentinel;
        for(int k=0;k<ldq*(cols?cols:1);k++)q[k]=sentinel;
        for(int j=0;j<cols;j++)for(int i=0;i<rows;i++)a[(size_t)j*lda+i]=creal(entry(global_index(i,block,prow,np_rows),global_index(j,block,pcol,np_cols),0));
        if(solver==1) solve_evp_real_(&n,&nev,a,&lda,ev,q,&ldq,&block,&fr,&fc);
        else elpa_solve_evp_real_2stage_double_impl_(&n,&nev,a,&lda,ev,q,&ldq,&block,&cols,&fr,&fc,&fp);
        for(int j=0;j<cols;j++) {
            for(int i=rows;i<ldq;i++)CHECK(q[(size_t)j*ldq+i]==sentinel);
            for(int i=rows;i<lda;i++)CHECK(a[(size_t)j*lda+i]==sentinel);
            int gj=global_index(j,block,pcol,np_cols);
            if(gj<nev)for(int i=0;i<rows;i++)vectors[(size_t)gj*n+global_index(i,block,prow,np_rows)]=q[(size_t)j*ldq+i];
        }
        free(a);free(q);
    }
    MPI_Allreduce(MPI_IN_PLACE,vectors,n*n,MPI_C_DOUBLE_COMPLEX,MPI_SUM,parent);
    double eigen_error=0,residual=0,orthogonality=0;
    for(int k=0;k<nev;k++) {
        CHECK(isfinite(ev[k]));
        for(int i=0;i<n;i++)CHECK(isfinite(creal(vectors[(size_t)k*n+i])) && isfinite(cimag(vectors[(size_t)k*n+i])));
        eigen_error=fmax(eigen_error,fabs(ev[k]-reference[k]));
        for(int i=0;i<n;i++) {
            double complex r=-ev[k]*vectors[(size_t)k*n+i];
            for(int j=0;j<n;j++)r+=entry(i,j,is_complex)*vectors[(size_t)k*n+j];
            residual=fmax(residual,cabs(r));
        }
        for(int j=0;j<nev;j++) {
            double complex dot=k==j?-1.0:0.0;
            for(int i=0;i<n;i++)dot+=conj(vectors[(size_t)k*n+i])*vectors[(size_t)j*n+i];
            orthogonality=fmax(orthogonality,cabs(dot));
        }
    }
    if(parent_rank==0)printf("CPU ELPA%d %s grid=%dx%d n=%d nev=%d: eigen=%.3g residual=%.3g orthogonality=%.3g\n",solver,is_complex?"complex":"real",np_rows,np_cols,n,nev,eigen_error,residual,orthogonality);
    CHECK(isfinite(eigen_error) && eigen_error<2e-11);
    CHECK(isfinite(residual) && residual<2e-11);
    CHECK(isfinite(orthogonality) && orthogonality<2e-11);
    free(ev);free(reference);free(full);free(vectors);
    MPI_Comm_free(&row_comm);MPI_Comm_free(&col_comm);
    ++checked;
}

int main(int argc,char **argv)
{
    int world_size,provided;
    CHECK(MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided)==MPI_SUCCESS);
    CHECK(provided>=MPI_THREAD_SERIALIZED);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    CHECK(world_size==6 || world_size==8);
    const int wide=world_size/2;
    for(int solver=1;solver<=2;solver++)for(int z=0;z<=1;z++) {
        run_case(MPI_COMM_WORLD,2,wide,solver,z,23,13,3);
        run_case(MPI_COMM_WORLD,wide,2,solver,z,11,11,2);
        run_case(MPI_COMM_WORLD,wide,2,solver,z,3,2,2);
        run_case(MPI_COMM_WORLD,2,wide,solver,z,3,2,2);
        /* Enough rows for both OpenMP workers and the SIMD backtransform. */
        run_case(MPI_COMM_WORLD,2,wide,solver,z,257,31,16);
    }
    /* Equal-sized, interleaved groups use different spectra and row/column
       orientations. Accidental MPI_COMM_WORLD collectives then cannot pass. */
    MPI_Comm group;
    int color=world_rank%2;
    MPI_Comm_split(MPI_COMM_WORLD,color,world_rank,&group);
    for(int solver=1;solver<=2;solver++)for(int z=0;z<=1;z++)
        run_case(group,color?wide:1,color?1:wide,solver,z,17+2*color,7+color,2);
    MPI_Comm_free(&group);
    if(world_rank==0)printf("PASS: %d CPU ELPA ABI cases per rank\n",checked);
    MPI_Finalize();return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-source', type=Path, default=ROOT / 'build/cuda/native/source')
    parser.add_argument('--mpiexec', default=os.environ.get('MPIEXEC', 'mpirun'))
    parser.add_argument('--make', default='make')
    parser.add_argument('--makefile', default='Makefile', help='Native site configuration, e.g. Makefile.kugui')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--ranks', type=int, choices=(6, 8), default=6)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    native = args.native_source.resolve()
    if not (native / 'Makefile').is_file():
        parser.error('Build OpenMX first, or provide --native-source')
    with tempfile.TemporaryDirectory(prefix='openmx-elpa-cpu-') as temporary:
        directory = Path(temporary)
        driver = directory / 'driver.c'
        driver.write_text(DRIVER)
        binary = directory / 'check'
        makefile = directory / 'check.mk'
        # Use exactly the production compiler and release library stack.
        makefile.write_text(
            '.PHONY: check-elpa-cpu\ncheck-elpa-cpu:\n'
            '\t$(CC_NOACC_O3) -I$(GPUSOLVER2_ELPA_INCLUDE) '
            f'-I{shlex.quote(str(ROOT / "source"))} '
            f'{shlex.quote(str(ROOT / "source/openmx_elpa_cpu.c"))} '
            f'{shlex.quote(str(ROOT / "source/openmx_env.c"))} '   # the kernel override is read through the registry
            f'{shlex.quote(str(driver))} -o {shlex.quote(str(binary))} $(LIB)\n')
        subprocess.run([args.make, '-C', str(native), '-f', args.makefile, '-f', str(makefile),
                        'check-elpa-cpu'], check=True, timeout=args.timeout)
        env = os.environ.copy()
        env.update(OPENMX_GPU='0', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', OPENBLAS_NUM_THREADS=str(args.threads), BLIS_NUM_THREADS=str(args.threads))
        subprocess.run([*shlex.split(args.mpiexec), '-np', str(args.ranks), '--map-by', f'slot:PE={args.threads}', '--bind-to', 'core', str(binary)],
                       check=True, timeout=args.timeout, env=env)


if __name__ == '__main__':
    main()

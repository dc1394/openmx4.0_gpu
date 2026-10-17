#!/usr/bin/env python3
"""Check the actual NC DC-LNO MI300A guard and CPU/GPU dispatch without HIP.

Extract the production helper and the real admission/dispatch block through the
CPU fallback condition. Solver bodies are stubs: these tests check that rejected
matrices reach the complete CPU local solve untouched, not eigenpair accuracy.
The existing run_dclno_noncol_retry.py checks the solver algebra and retries.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def function(source, name):
    match = re.search(r'^static [^;{}]*\b'+re.escape(name)+r'\([^;]*?\)\s*\{', source, re.M)
    if not match:
        raise ValueError('Cannot find production function '+name)
    start = source.index('{',match.start())
    depth, end = 1, start+1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}')
        end += 1
    return source[match.start():end]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <float.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { double r, i; } dcomplex;
static int checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while(0)
static int mi300a, queries;
int openmx_gpu_is_mi300a(void) { queries++; return mi300a; }
'''

DISPATCH_STUBS = r'''
#define measure_time 0
#define DCLNO_RETRY_NO_MEMORY INT_MAX
static int use_gpu_accel_nc, gpu_no_memory_nc, Eigen_MPI_flag, threshold;
static int SCF_iter=1, Gc_AN=13, myid0=0, Mc_AN=0, level_stdout=0;
static int NUM=4, NUMH=2, LNO_recalc_flag=1, HO_TC[1]={1};
static int gpu_calls, cpu_calls, fault, fatal_calls;
static dcomplex overlap[4], hamiltonian[16], saved_s[4], saved_h[16];
static dcomplex *BLAS_OLP=overlap, *BLAS_H=hamiltonian;
static double ko[5];
static jmp_buf fatal_jump;
static int DCLNO_NonColGpuEigenThreshold(void) { return threshold; }
static void dtime(double *x) { *x=0; }
static void DCLNO_AbortWithMessage(const char *message) { (void)message; fatal_calls++; longjmp(fatal_jump,1); }
static int DCLNO_Solve_NonCol_GpuSolver(int n, int ns, int count, int rebuild,
                                       dcomplex *s, dcomplex *h, double *w)
{
    (void)w;
    CHECK(n==NUM && ns==NUMH && count==NUM && rebuild==1);
    CHECK(!memcmp(s,saved_s,sizeof saved_s) && !memcmp(h,saved_h,sizeof saved_h));
    gpu_calls++;
    return fault;
}
static void cpu_local_solve(void)
{
    CHECK(!memcmp(BLAS_OLP,saved_s,sizeof saved_s) && !memcmp(BLAS_H,saved_h,sizeof saved_h));
    cpu_calls++;
}
static void snapshot(void)
{
    memcpy(saved_s,overlap,sizeof overlap);
    memcpy(saved_h,hamiltonian,sizeof hamiltonian);
}
static void reset(void)
{
    memset(overlap,0,sizeof overlap); memset(hamiltonian,0,sizeof hamiltonian);
    overlap[0].r=2; overlap[3].r=3;
    overlap[1]=(dcomplex){0.1,-0.2}; overlap[2]=(dcomplex){0.1,0.2};
    for (int i=0;i<4;i++) hamiltonian[5*i].r=i+1;
    hamiltonian[1]=(dcomplex){0.3,-0.4}; hamiltonian[4]=(dcomplex){0.3,0.4};
    mi300a=1; queries=gpu_calls=cpu_calls=fatal_calls=fault=0;
    use_gpu_accel_nc=1; gpu_no_memory_nc=Eigen_MPI_flag=0; threshold=1;
    snapshot();
}
static void dispatch(void)
{
    int use_gpu_task_nc, gpu_retry_info_nc, NUM2, nc_guard_reported=0;
    double stime=0, etime=0, time3=0;
'''

TESTS = r'''
static uint64_t rng=0xd11238415;
static uint64_t rnd(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static double uni(void){return (double)(rnd()>>11)*0x1p-53;}
static int numerical_checks=0;
static void compare(const dcomplex *a,int n) {
    DCLNO_NCGuardStats x,y;
    int p=DCLNO_NCGuardMatrixSlow(a,n,&x),q=DCLNO_NCGuardMatrix(a,n,&y);
    if(p!=q || (p==2 && (x.max_abs!=y.max_abs || x.max_anti!=y.max_anti)) || (p!=2 && (fabs(x.max_abs-y.max_abs)>8*DBL_EPSILON*fmax(DBL_MIN,x.max_abs) ||
                       fabs(x.max_anti-y.max_anti)>8*DBL_EPSILON*fmax(DBL_MIN,x.max_anti)))) {
        fprintf(stderr,"DIFFERENCE case=%d n=%d status=%d/%d abs=%.17g/%.17g anti=%.17g/%.17g\n",
                numerical_checks,n,p,q,x.max_abs,y.max_abs,x.max_anti,y.max_anti);exit(3);
    }
    numerical_checks++;
}
static void check_numerical_equivalence(void) {
    dcomplex a[49];
    compare(NULL,1);compare(a,0);compare(a,2147483647);
    const double edge[]={0,DBL_TRUE_MIN,DBL_MIN,1e-151,1e-150,1e-149,1e149,1e150,1e151,DBL_MAX,INFINITY,NAN};
    for(int j=0;j<12;j++)for(int k=0;k<12;k++)for(int t=0;t<2;t++) {
        memset(a,0,sizeof a);a[0].r=edge[j];a[3].r=edge[k];a[2].r=t?edge[j]:-edge[j];
        a[1].r=edge[k];a[1].i=edge[j];a[2].i=-edge[k];compare(a,2);
    }

    /* An invalid late element must retain the original partial statistics. */
    for(int kind=0;kind<3;kind++) {
        memset(a,0,sizeof a);a[0]=(dcomplex){3,1e-9};
        a[48]=(dcomplex){2,kind==0?NAN:kind==1?INFINITY:DBL_MAX};
        compare(a,7);
    }
    for(int shape=0;shape<4;shape++) {
        memset(a,0,sizeof a);
        for(int j=0;j<7;j++)for(int i=0;i<=j;i++) {
            double r=(double)(i+j+1)/17, v=(double)(j-i)/31;
            if(shape==1)v=0;
            if(shape==2 && i!=j)r=v=0;
            if(shape==3)r=v=0;
            a[j*7+i]=(dcomplex){r,v};a[i*7+j]=(dcomplex){r,-v};
        }
        compare(a,7);
        a[0].i=1e-6;compare(a,7);
    }
    for(int t=0;t<40000;t++) {
        int n=t%3==0?1:t%3==1?2:7;
        double scale=t%4?ldexp(1.0,(int)(rnd()%80)-40):ldexp(1.0,(int)(rnd()%2046)-1022);
        for(int j=0;j<n;j++)for(int i=0;i<=j;i++) {
            double r=scale*(2*uni()-1),v=i==j?0:scale*(2*uni()-1);
            a[j*n+i]=(dcomplex){r,v};a[i*n+j]=(dcomplex){r,-v};
        }
        if(t%3)a[0].i=scale*(t%2?1e-9:1e-6);
        compare(a,n);
    }
    for(int p=-500;p<=500;p+=5)for(int q=-64;q<=64;q++) {
        double scale=ldexp(1.0,p),limit=1e-8*fmax(1.0,scale);
        double d=limit/2;
        for(int k=0;k<abs(q);k++)d=nextafter(d,q<0?0:INFINITY);
        a[0]=(dcomplex){scale,d};compare(a,1);
    }
    printf("PASS: %d fast/slow numerical cases; boundary replay, finite/extreme/invalid inputs\n",numerical_checks);
}
int main(void)
{
    DCLNO_NCGuardStats stats;
    check_numerical_equivalence();
    reset();
    CHECK(DCLNO_NCGuardMatrix(overlap,2,&stats)==0 && stats.max_anti==0);
    dispatch(); CHECK(gpu_calls==1 && cpu_calls==0 && queries==1);
    reset(); hamiltonian[4].r+=1e-6; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==1 && !gpu_no_memory_nc);
    reset(); overlap[2].r+=1e-6; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==1);
    reset(); hamiltonian[0].i=3e-8; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1);
    reset(); hamiltonian[0].i=1e-9; snapshot();
    dispatch(); CHECK(gpu_calls==1 && cpu_calls==0);
    /* Even poisoned matrices must not be scanned for CPU/Radeon/skipped tasks. */
    reset(); use_gpu_accel_nc=0; overlap[0].i=NAN; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==0 && fatal_calls==0);
    reset(); mi300a=0; overlap[0].i=NAN; snapshot();
    dispatch(); CHECK(gpu_calls==1 && cpu_calls==0 && queries==1 && fatal_calls==0);
    reset(); threshold=5; overlap[0].i=NAN; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==0 && fatal_calls==0);
    reset(); gpu_no_memory_nc=1; overlap[0].i=NAN; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==0 && fatal_calls==0);
    reset(); Eigen_MPI_flag=1; overlap[0].i=NAN; snapshot();
    dispatch(); CHECK(gpu_calls==0 && cpu_calls==1 && queries==0 && fatal_calls==0);
    /* Existing retry and no-memory admission remain intact. */
    reset(); fault=1; dispatch(); CHECK(gpu_calls==1 && cpu_calls==1 && !gpu_no_memory_nc);
    reset(); fault=DCLNO_RETRY_NO_MEMORY; dispatch();
    CHECK(gpu_calls==1 && cpu_calls==1 && gpu_no_memory_nc);
    dispatch(); CHECK(gpu_calls==1 && cpu_calls==2 && queries==1);
    /* Invalid admitted MI input never reaches either numerical solver. */
    reset(); hamiltonian[1].i=NAN; snapshot();
    if (!setjmp(fatal_jump)) { dispatch(); CHECK(0); }
    CHECK(fatal_calls==1 && gpu_calls==0 && cpu_calls==0);
    CHECK(!memcmp(hamiltonian,saved_h,sizeof saved_h));
    reset(); overlap[0].i=INFINITY;
    if (!setjmp(fatal_jump)) { dispatch(); CHECK(0); }
    CHECK(fatal_calls==1 && gpu_calls==0 && cpu_calls==0);
    CHECK(DCLNO_NCGuardMatrix(NULL,2,&stats)==2);
    CHECK(DCLNO_NCGuardMatrix(overlap,0,&stats)==2);
    CHECK(DCLNO_NCGuardMatrix(overlap,INT_MAX,&stats)==2);
    overlap[0]=(dcomplex){0,DBL_MAX};
    CHECK(DCLNO_NCGuardMatrix(overlap,1,&stats)==2);
    overlap[0]=(dcomplex){0,4e-9};
    CHECK(DCLNO_NCGuardMatrix(overlap,1,&stats)==0);
    overlap[0].i=6e-9;
    CHECK(DCLNO_NCGuardMatrix(overlap,1,&stats)==1);
    overlap[0]=(dcomplex){1e6,0.004};
    CHECK(DCLNO_NCGuardMatrix(overlap,1,&stats)==0);
    overlap[0].i=0.006;
    CHECK(DCLNO_NCGuardMatrix(overlap,1,&stats)==1);
    CHECK(DCLNO_NCGuardMatrix((dcomplex[]){{DBL_MAX,0},{-DBL_MAX,0},{DBL_MAX,0},{0,0}},2,&stats)==2);
    /* Explicit GPU verbosity and ordinary output verbosity both report fallback. */
    reset(); level_stdout=1; CHECK(setenv("OPENMX_GPU_VERBOSE","1",1)==0);
    hamiltonian[4].r+=1e-6; snapshot(); dispatch();
    CHECK(gpu_calls==0 && cpu_calls==1);
    reset(); level_stdout=1; CHECK(setenv("OPENMX_GPU_VERBOSE","0",1)==0);
    hamiltonian[4].r+=1e-6; snapshot(); dispatch();
    CHECK(gpu_calls==0 && cpu_calls==1);
    reset(); level_stdout=2;
    hamiltonian[4].r+=1e-6; snapshot(); dispatch();
    CHECK(gpu_calls==0 && cpu_calls==1);
    printf("PASS: %d host checks; actual MI guard/admission/GPU-retry/whole-CPU-fallback dispatch, immutable S/H, finite/overflow/scale\n",checks);
    return 0;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'source/Divide_Conquer_LNO.c')
    parser.add_argument('--sanitize',action='store_true')
    parser.add_argument('--cc',default=os.environ.get('CC','cc'))
    args=parser.parse_args()
    source=args.source.read_text()
    typedef=re.search(r'typedef struct\s*\{\s*double max_abs, max_anti;\s*\}\s*DCLNO_NCGuardStats;',source).group(0)
    helper=typedef+'\n'+function(source,'DCLNO_NCGuardMatrixSlow')+'\n'+function(source,'DCLNO_NCGuardMatrix')+'\n'+function(source,'DCLNO_NCGuardProblem')
    start=source.index('    use_gpu_task_nc = (use_gpu_accel_nc')
    cpu_entry='    if (!use_gpu_task_nc || 0 < gpu_retry_info_nc){'
    end=source.index(cpu_entry,start)+len(cpu_entry)
    dispatch=source[start:end]+'\n      cpu_local_solve();\n    }\n}\n'
    test=PREFIX+helper+DISPATCH_STUBS+dispatch+TESTS
    with tempfile.TemporaryDirectory(prefix='dclno_mi_guard_') as tmp:
        c=Path(tmp)/'test.c'; exe=Path(tmp)/'test'; c.write_text(test)
        flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        subprocess.run([args.cc,*flags,'-std=c11','-Wall','-Wextra',str(c),'-lm','-o',str(exe)],check=True)
        env=dict(os.environ); env.pop('OPENMX_GPU_VERBOSE',None)
        result=subprocess.run([str(exe)],env=env,text=True,capture_output=True,check=True)
        assert result.stderr.count('DCLNO_NC_GUARD ')==2, result.stderr
        print(result.stdout,end='')
        print('PASS: explicit GPU verbosity and ordinary verbosity report fallback; quiet mode stays quiet.')


if __name__=='__main__':
    main()

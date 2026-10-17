#!/usr/bin/env python3
"""Regression for first/last radial intervals and points outside the PAO mesh.

Exercises production scalar, contracted, derivative, reusable-workspace,
Set_Orbitals_Grid's two CPU inlines, Force3's CPU derivative inline, and
the Voronoi orbital evaluator. The HIP evaluators are covered separately
by run_orbs_grid_boundaries.py. Guard NaNs and poisoned scratch reveal
premature boundary reads and uninitialized angular data. Values and gradients
are compared with an independent Hermite reference. CC/CFLAGS and --valgrind
are supported.
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index(name)
    pos = source.index("{", start)
    depth, end = 1, pos + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


source = (ROOT / "source/Set_Orbitals_Grid.c").read_text()
functions = ""
for filename in ("Get_Orbitals.c", "Get_Cnt_Orbitals.c", "Get_dOrbitals.c", "Get_Cnt_dOrbitals.c"):
    text = (ROOT / "source" / filename).read_text()
    functions += "\n" + text[text.index("void Get_"):]

# Extract both actual inlined CPU evaluators; their surrounding grid iteration
# and MPI transfers do not participate in the spline calculation.
inline_start = source.index("double Set_Orbitals_Grid(")
for index in range(2):
    inline_start = source.index("/* start of inlining of Get_Orbitals */", inline_start)
    inline_end = source.index("/* end of inlining of Get_Orbitals */", inline_start)
    inline = source[inline_start:inline_end]
    inline_start = inline_end
    functions += f"\nstatic void inline_{index}(double x, double y, double z, double *Chi0) {{\n" + r'''
    int Cwan=0,Hwan=0,wan,i,j,L0,Mul0,M0,i1,m,mp_min,mp_max,po;
    double dum,dum1,dum2,dum3,dum4,a,b,c,d,siQ,coQ,siP,coP,Q,P,R;
    double r,r1,Min_r,theta,phi,rm,df,sum0,sum1;
    double h1,h2,h3,f1,f2,f3,f4,g1,g2,x1,x2,y1,y2,y12,y22,f;
    double RF[4][2],AF[4][7],SH[7][2],dSHt[7][2],dSHp[7][2];
    for(i=0;i<4;i++) for(j=0;j<7;j++) AF[i][j]=NAN;
''' + inline + "\n}\n"

force = (ROOT / "source/Force.c").read_text()
force_start = force.index("/* start: direct inlining of Get_dOrbitals_work */")
force_end = force.index("/* end: direct inlining of Get_dOrbitals_work */", force_start)
functions += "\nstatic void inline_force(int Cwan, double dx, double dy, double dz, " \
    "double **dorbs0, struct WORK_DORBITAL work_dObs) {\n{\n" \
    + force[force_start:force_end] + "\n}\n}\n"
voronoi = (ROOT / "source/Voronoi_Orbital_Moment.c").read_text()
functions += "\n" + function(voronoi[voronoi.index("void Voronoi_Orbital_Moment"):],
                             "void Orbitals_on_Grid(")

PRELUDE = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PI 3.141592653589793238462643383279502884
#define Supported_MaxL 3
#define SOG_L0MAX 3
#define SOG_MULMAX 8
struct WORK_DORBITAL { double **RF, **dRF, **AF, **dAFQ, **dAFP; };
typedef struct { double r,i; } dcomplex;
static dcomplex Comp2Real[4][7][7];
static int List_YOUSO[64], M2G[2]={0,1}, F_M2G[4]={0,1,1,1}, WhatSpecies[2]={0,0};
static int Spe_Num_Mesh_PAO[1]={6}, Spe_MaxL_Basis[1]={0};
static int Spe_PAO_LMAX[1]={0}, Spe_PAO_Mul[1]={2};
static int nb[4]={2}, *Spe_Num_Basis[1]={nb}, cb[4]={1}, *Spe_Num_CBasis[1]={cb};
static int specified[2]={2,0}, *Spe_Specified_Num[1]={specified};
static double coeff[2]={0.75,-0.25}, *ca[1]={coeff}, **CntCoes[4]={ca,ca,ca,ca};
static double radius[8]={NAN,0.1,0.2,0.4,0.8,1.6,3.2,NAN};
static double wave0[8]={NAN,1.1,-0.3,0.7,1.5,-0.2,0.9,NAN};
static double wave1[8]={NAN,-0.2,0.8,1.2,-0.4,0.6,-0.7,NAN};
static double *Spe_PAO_RV[1]={radius+1};
static double *mul[2]={wave0+1,wave1+1}, **angular[1]={mul}, ***Spe_PAO_RWF[1]={angular};
static double sgn(double x) { return x < 0.0 ? -1.0 : 1.0; }
static void xyz2spherical(double x,double y,double z,double ox,double oy,double oz,double *out)
{
    x-=ox; y-=oy; z-=oz;
    out[0]=sqrt(x*x+y*y+z*z);
    out[1]=out[0] ? acos(z/out[0]) : PI/2;
    out[2]=atan2(y,x);
}
static void ComplexSH(int l,int m,double q,double p,double *s,double *dq,double *dp)
{ (void)l;(void)m;(void)q;(void)p;(void)s;(void)dq;(void)dp; abort(); }
static void *poison_malloc(size_t n)
{
    void *p=malloc(n); assert(p); memset(p,0xff,n); return p;
}
#define malloc poison_malloc
'''

HARNESS = r'''
#undef malloc
static void radial_reference(double r,int orbital,double *value,double *gradient)
{
    const double *rv=radius+1,*rw=mul[orbital];
    if(r>rv[5]) { *value=*gradient=0.0; return; }
    int m=1;
    while(m<5 && rv[m]<r) ++m;
    const double hm=m==1 ? rv[0]-rv[2] : rv[m-1]-rv[m-2];
    const double h=rv[m]-rv[m-1], hp=m==5 ? rv[3]-rv[5] : rv[m+1]-rv[m];
    const double ym=rw[m==1 ? m+1 : m-2], y0=rw[m-1], y1=rw[m];
    const double yp=rw[m==5 ? m-2 : m+1];
    const double d0=((y1-y0)*hm/h+(y0-ym)*h/hm)/(hm+h);
    const double d1=((yp-y1)*h/hp+(y1-y0)*hp/h)/(h+hp);
    const double t=(r-rv[m-1])/h, t2=t*t, t3=t2*t;
    const double angular=0.282094791773878;
    *value=angular*((2*t3-3*t2+1)*y0+(t3-2*t2+t)*h*d0+(-2*t3+3*t2)*y1+(t3-t2)*h*d1);
    *gradient=angular*((6*t2-6*t)*y0/h+(3*t2-4*t+1)*d0+(-6*t2+6*t)*y1/h+(3*t2-2*t)*d1);
}
static void equal(double actual,double expected)
{
    if(!isfinite(actual) || fabs(actual-expected)>1e-12*(1+fabs(expected))) {
        fprintf(stderr,"orbital mismatch: %.17g != %.17g\n",actual,expected); abort();
    }
}
static void check_derivative(double **d,double *v,double *g,int sign,int count)
{
    for(int i=0;i<count;i++) {
        equal(d[0][i],v[i]); equal(d[1][i],sign*g[i]);
        equal(d[2][i],0.0); equal(d[3][i],0.0);
    }
}
int main(void)
{
    List_YOUSO[7]=2; List_YOUSO[24]=2; List_YOUSO[25]=0;
    struct WORK_DORBITAL work;
    Get_dOrbitals_init(&work);
    const double radii[]={0.1,0.11,0.15,0.2,0.3,0.4,0.6,0.8,1.2,1.6,2.0,3.1,3.2,3.3};
    double dmem[4][2],*d[4]={dmem[0],dmem[1],dmem[2],dmem[3]},v[2],g[2],chi[2];
    double vrad[2],vang[1],vchi[2],*vr[1]={vrad},*va[1]={vang};
    double *vc_m[1]={vchi},**vc[1]={vc_m};
    for(int k=0;k<(int)(sizeof(radii)/sizeof(radii[0]));k++) {
        const double r=radii[k];
        for(int i=0;i<2;i++) radial_reference(r,i,&v[i],&g[i]);
        Get_Orbitals(0,r,0,0,chi); for(int i=0;i<2;i++) equal(chi[i],v[i]);
        inline_0(r,0,0,chi); for(int i=0;i<2;i++) equal(chi[i],v[i]);
        inline_1(r,0,0,chi); for(int i=0;i<2;i++) equal(chi[i],v[i]);
        Get_dOrbitals(0,r,0,0,d); check_derivative(d,v,g,-1,2);
        Get_dOrbitals_work(0,r,0,0,d,work); check_derivative(d,v,g,-1,2);
        if(r>radius[6]) work.AF[0][0]=work.dAFQ[0][0]=work.dAFP[0][0]=NAN;
        inline_force(0,r,0,0,d,work);
        for(int i=0;i<2;i++) {
            equal(d[1][i],-g[i]); equal(d[2][i],0.0); equal(d[3][i],0.0);
        }
        Get_dOrbitals2(0,r,0,0,d); check_derivative(d,v,g,1,2);
        vang[0]=NAN;
        Orbitals_on_Grid(0,0,2,r,0,0,vc,vr,va);
        for(int i=0;i<2;i++) equal(vchi[i],v[i]);
        const double cv=coeff[0]*v[0]+coeff[1]*v[1],cg=coeff[0]*g[0]+coeff[1]*g[1];
        Get_Cnt_Orbitals(1,r,0,0,chi); equal(chi[0],cv);
        v[0]=cv; g[0]=cg;
        Get_Cnt_dOrbitals(1,r,0,0,d); check_derivative(d,v,g,-1,1);
        Get_Cnt_dOrbitals2(1,r,0,0,d); check_derivative(d,v,g,1,1);
        /* Halo indices exceed the rank-local M2G array. */
        Get_Cnt_Orbitals(3,r,0,0,chi); equal(chi[0],cv);
        Get_Cnt_dOrbitals(3,r,0,0,d); check_derivative(d,v,g,-1,1);
        Get_Cnt_dOrbitals2(3,r,0,0,d); check_derivative(d,v,g,1,1);
    }
    Get_dOrbitals_free(work);
    puts("PASS: 196 orbital spline cases, CPU FP64, Force3 inline, Voronoi, contracted halo values, gradients and out-of-mesh zeroes");
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-orbital-spline-") as tmp:
    src, exe = Path(tmp)/"check.c", Path(tmp)/"check"
    src.write_text(PRELUDE+functions+HARNESS)
    cc = shlex.split(os.environ.get("CC","cc"))
    flags = shlex.split(os.environ.get("CFLAGS","-O2 -g"))
    subprocess.run(cc+flags+["-std=c11",str(src),"-lm","-o",str(exe)],check=True)
    runner = ["valgrind","--error-exitcode=99","--quiet","--leak-check=full"] if "--valgrind" in sys.argv[1:] else []
    subprocess.run(runner+[str(exe)],check=True)

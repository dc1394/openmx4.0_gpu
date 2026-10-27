#!/usr/bin/env python3
"""Compare real orbital evaluators against frozen code and spatial derivatives.

The production functions run with deliberately inconsistent legacy metadata;
all PAO mesh, multiplicity and basis information must come from their views.
ASan/UBSan cover origin/interior/endpoints/outside, axes, g orbitals, multiple
species, contraction and reusable scratch. No CUDA/HIP runtime is initialized.
"""
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    # These helper functions contain no braces in string literals/comments.
    hit = re.search(r'^(?:void|double)\s+' + name + r'\([^;]*?\)\s*\{', source, re.M)
    start = source.index('{', hit.start())
    end, depth = start + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[hit.start():end]


PREFIX = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_core_atomic.h"
#include "openmx_core_local_basis.h"
#include "openmx_complex.h"
#include "openmx_basis_harmonics.h"
#define PI 3.1415926535897932384626
#define Supported_MaxL 4
#define YOUSO36 4
#define sgn(x) ((x)<0 ? -1. : 1.)
static int List_YOUSO[60],*Spe_Num_Mesh_PAO,*Spe_MaxL_Basis,**Spe_Num_Basis,**Spe_Num_CBasis;
static int *F_M2G,*WhatSpecies,**Spe_Specified_Num;
static double **Spe_PAO_RV,****Spe_PAO_RWF,***CntCoes;
static dcomplex Comp2Real[5][11][11];
static int mpi_comm_level1;
struct WORK_DORBITAL { double **RF,**dRF,**AF,**dAFQ,**dAFP; };
static dcomplex Complex(double r,double i) { return (dcomplex){r,i}; }
void Associated_Legendre(int l,int m,double x,double ALeg[2]);
'''

library = Path(sys.argv[1]).resolve(strict=True)
with tempfile.TemporaryDirectory(prefix='openmx-orbital-evaluation-') as directory:
    folder = Path(directory)
    (folder/'mpi.h').write_text('#pragma once\n#include <stdlib.h>\n'
                              'static inline int MPI_Abort(int comm,int code) {(void)comm;(void)code;abort();}\n')
    helpers = (ROOT/'source/openmx_common.c').read_text()
    parts = [PREFIX, (ROOT/'source/xyz2spherical.c').read_text().replace('#include "openmx_common.h"', ''),
             function(helpers, 'Associated_Legendre'), function(helpers, 'ComplexSH'),
             (ROOT/'modules/basis/tests/fixtures/real_harmonics_before_owner.inc').read_text(),
             (ROOT/'tests/fixtures/orbitals_before_flat_views.c').read_text()]
    for name in ('Get_Orbitals.c', 'Get_dOrbitals.c', 'Get_Cnt_Orbitals.c', 'Get_Cnt_dOrbitals.c'):
        parts.append((ROOT/'source'/name).read_text().replace('#include "openmx_common.h"', ''))
    # Compile the actual host fallbacks from both backend source selections.
    # They have their own inlined copies, so testing the scalar function alone
    # cannot detect a missed correction in Force or orbital grid construction.
    for backend, directory in [('cuda', ROOT/'source'), ('hip', ROOT/'source/backends/hip/src')]:
        force = (directory/'Force.c').read_text()
        begin = force.index('/* start: direct inlining of Get_dOrbitals_work */')
        end = force.index('/* end: direct inlining of Get_dOrbitals_work */', begin)
        parts.append(f'static void force_{backend}(const OpenMXAtomicViews *atomic, int Cwan, '
                     'double dx, double dy, double dz, double **dorbs0, OpenMXOrbitalBuffers work_dObs) {\n{\n'
                     + force[begin:end] + '\n}\n}\n')
        grid = (directory/'Set_Orbitals_Grid.c').read_text()
        end = grid.index('double Set_Orbitals_Grid(')
        for index in range(2):
            begin = grid.index('/* start of inlining of Get_Orbitals */', end)
            end = grid.index('/* end of inlining of Get_Orbitals */', begin)
            parts.append(f'static void grid_{backend}_{index}(const OpenMXAtomicViews *atomic, '
                         'int species, double x, double y, double z, double *Chi0) {\n' + r'''
    int Cwan=species,Hwan=species,wan,i,j,L0,Mul0,M0,i1,m,mp_min,mp_max,po;
    double dum,dum1,dum2,dum3,dum4,a,b,c,d,siQ,coQ,siP,coP,Q,P,R;
    double r,r1,Min_r,theta,phi,rm,df,sum0,sum1;
    double h1,h2,h3,f1,f2,f3,f4,g1,g2,x1,x2,y1,y2,y12,y22,f;
    double RF[15]={0},AF[55]={0},SH[9][2],dSHt[9][2],dSHp[9][2];
    OpenMXOrbitalBuffers orbital_buffers={5,3,11,RF,NULL,AF,NULL,NULL};
''' + grid[begin:end] + '\n}\n')
    parts.append((ROOT/'tests/orbital_evaluation_harness.c').read_text())
    source = folder/'check.c'
    source.write_text('\n'.join(parts))
    command = shlex.split(os.environ.get('CC', 'cc')) + [
        '-std=c11', '-O1', '-g', '-fno-pie', '-no-pie', '-ffp-contract=off',
        '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
        '-I'+str(folder), '-I'+str(ROOT/'source'), str(source), str(library),
        '-Wl,-rpath,'+str(library.parent), '-lm', '-o', str(folder/'check')]
    subprocess.run(command, check=True)

    def bounded_stack():
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (8*1024*1024, hard))

    subprocess.run([str(folder/'check')], check=True, timeout=90, preexec_fn=bounded_stack)

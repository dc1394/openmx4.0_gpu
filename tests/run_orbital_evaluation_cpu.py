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
#include "openmx_complex.h"
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
             function((ROOT/'source/SetPara_DFT.c').read_text(), 'Set_Comp2Real'),
             (ROOT/'tests/fixtures/orbitals_before_flat_views.c').read_text()]
    for name in ('Get_Orbitals.c', 'Get_dOrbitals.c', 'Get_Cnt_Orbitals.c', 'Get_Cnt_dOrbitals.c'):
        parts.append((ROOT/'source'/name).read_text().replace('#include "openmx_common.h"', ''))
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

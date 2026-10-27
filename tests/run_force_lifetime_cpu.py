#!/usr/bin/env python3
"""Exercise both production Force entry points against the real C++ owner."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from native_state_test_support import bounded_stack

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(os.environ["OPENMX_CORE_LIBRARY"]).resolve(strict=True)


def function(source, declaration):
    begin = source.index(declaration)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


PREFIX = r'''
#include "openmx_core_force.h"
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {struct {const int *contracted_orbitals;} species;} OpenMXAtomicViews;
typedef struct Reciprocal OpenMXReciprocalViews;
typedef struct LocalBasis OpenMXLocalBasisViews;
typedef struct DensityGpuState OpenMXDensityGpuState;
typedef struct OpenMXHamiltonianGpuState OpenMXHamiltonianGpuState;
static int atomnum=3,Matomnum=1,MatomnumF=1,SpeciesNum=2,SpinP_switch;
static int SO_switch,Hub_U_switch,F_U_flag,Constraint_NCS_switch,Zeeman_NCS_switch,Zeeman_NCO_switch;
static int F_M2G[]={-1,3,1},WhatSpecies[]={-1,0,0,1},Spe_Total_CNO[]={3,1},FNAN[]={27,1,0,1};
static int n1[]={1,2},n2[]={2},n3[]={3,1};static int *natn[]={NULL,n1,n2,n3};
static int expected_imaginary,invocations;
static const OpenMXAtomicViews *expected_atomic;
static OpenMXDensityGpuState *expected_density_state;
static OpenMXHamiltonianGpuState *expected_hamiltonian_state;
static double *****expected_h0,******expected_ds,*****expected_olp,*****expected_cdm,*****expected_edm;
static void dtime(double *out){static double clock;*out=clock++;}
static void OpenMX_Solver_Failure(const char *error){fprintf(stderr,"%s\n",error);abort();}
static double OpenMX_ForceRun(const OpenMXForceInput *in,OpenMXForceOperation op,void *args){
  double result;char error[256];
  assert(in->imaginary_density==expected_imaginary);
  assert(openmx_core_force_run(in,omp_get_max_threads(),op,args,&result,error,sizeof(error)));
  assert(result==23);return result;
}
'''
STAGES = r'''
static int stage_calls;
static double OpenMX_Solver_Run(OpenMXSolverOperation op,void *args){
  double result;char error[256];
  assert(openmx_core_solver_run(omp_get_max_threads(),op,args,&result,error,sizeof(error)));
  assert(result==0);return result;
}
static void stage_scratch(OpenMXSolverInvocation *scratch){
  int *values;char error[256];
  assert(openmx_core_solver_int(scratch,0,"same-stage-name",29,&values,error,sizeof(error)));
  assert(values[28]==0);values[28]=++stage_calls;
}
static void Force3(OpenMXHamiltonianGpuState *h,const OpenMXAtomicViews *a,const OpenMXElectronicViews *e,
  const OpenMXLocalBasisViews *l,OpenMXSolverInvocation *s){
  assert(h==expected_hamiltonian_state && a==expected_atomic && !e && !l);stage_scratch(s);
}
static void Force4B(const OpenMXVnaMatrixViews *v,const OpenMXReciprocalViews *r,
  double *****d,const OpenMXLocalBasisViews *l,OpenMXSolverInvocation *s){
  assert(!v && !r && d==expected_cdm && !l);stage_scratch(s);
}
static void Force_HNL(const OpenMXAtomicViews *a,double *****d,double *****im,
  const OpenMXElectronicViews *e,const OpenMXLocalBasisViews *l,OpenMXSolverInvocation *s){
  assert(a==expected_atomic && d==expected_cdm && im==expected_edm && !e && !l);stage_scratch(s);
}
static void Force_CoreHole(const OpenMXAtomicViews *a,double *****d,double *****im,const OpenMXElectronicViews *e,OpenMXSolverInvocation *s){
  assert(a==expected_atomic && d==expected_cdm && im==expected_edm && !e);stage_scratch(s);
}
#include "openmx_force_stages.h"
'''
BODY = r'''
{
  assert(hamiltonian_state==expected_hamiltonian_state && density_state==expected_density_state);
  assert(atomic==expected_atomic && !vna && !reciprocal && !electronic && !local_basis);
  assert(H0==expected_h0 && DS_NL==expected_ds && OLP==expected_olp && CDM==expected_cdm && EDM==expected_edm);
  const OpenMXAtomBlockLayout *l=&force_work->density.layout;
  assert(l->atoms==3 && l->components==(size_t)SpinP_switch+1);
  assert(l->shapes[2*l->atom_offsets[1]]==1 && l->shapes[2*l->atom_offsets[2]]==3);
  assert(!!force_work->imaginary_rows==expected_imaginary);
  for(size_t i=0;i<l->elements;++i)assert(force_work->density.values[i]==0);
  force_work->density_rows[0][2][1][2][2]=7;
#pragma omp parallel
  {
    char error[256];const int worker=omp_get_thread_num();
    const size_t shape[]={3,5,7,1};OpenMXSolverComplexTensor values,again;
    assert(openmx_core_solver_complex(force_work->scratch,worker,"force-test",shape,&values,error,sizeof(error)));
    assert(values.values[104].r==0);values.values[104].r=worker+1;
    assert(openmx_core_solver_complex(force_work->scratch,worker,"force-test",shape,&again,error,sizeof(error)));
    assert(values.values==again.values && again.values[104].r==worker+1);
    OpenMXOrbitalBuffers sampler,again_sampler;
    OpenMX_Solver_Orbitals(force_work->scratch,"native-force-orbitals",4,3,1,&sampler);
    assert(sampler.angular_channels==5 && sampler.multiplicity==3 && sampler.magnetic_capacity==11);
    assert(sampler.dRF==sampler.RF+15 && sampler.dAFP==sampler.AF+110);
    assert(sampler.RF[14]==0 && sampler.dAFP[54]==0);
    sampler.RF[14]=worker+1;sampler.dAFP[54]=worker+3;
    OpenMX_Solver_Orbitals(force_work->scratch,"native-force-orbitals",4,3,1,&again_sampler);
    assert(sampler.RF==again_sampler.RF && again_sampler.dAFP[54]==worker+3);
  }
  assert(FNAN[0]==27);++invocations;return 23;
}
'''
MAIN = r'''
int main(void){
  int hamiltonian_token=0;expected_hamiltonian_state=(OpenMXHamiltonianGpuState*)&hamiltonian_token;
  int density_token=0;expected_density_state=(OpenMXDensityGpuState*)&density_token;
  OpenMXAtomicViews atomic={{Spe_Total_CNO}};expected_atomic=&atomic;
  double *addresses[5]={0};expected_h0=(double*****)&addresses[0];expected_ds=(double******)&addresses[1];
  expected_olp=(double*****)&addresses[2];expected_cdm=(double*****)&addresses[3];expected_edm=(double*****)&addresses[4];
  omp_set_dynamic(0);
  for(int threads=1;threads<=8;threads*=2){omp_set_num_threads(threads);
    for(int spin=0;spin<=3;spin+=(spin==1?2:1)){SpinP_switch=spin;
      for(int mode=0;mode<7;++mode){
        SO_switch=mode==1;Hub_U_switch=F_U_flag=mode==2;Constraint_NCS_switch=mode==3;
        Zeeman_NCS_switch=mode==4;Zeeman_NCO_switch=mode==5;
        expected_imaginary=(mode==1 || (mode==2 && spin==3) || (mode>=3 && mode<=5));
        assert(Force(expected_hamiltonian_state,expected_density_state,&atomic,NULL,NULL,expected_h0,expected_ds,expected_olp,expected_cdm,expected_edm,NULL,NULL)==1);
      }
    }
  }
  for(int i=0;i<3;++i){
    OpenMX_ForceGrid(expected_hamiltonian_state,&atomic,NULL,NULL);
    OpenMX_ForceVna(NULL,NULL,expected_cdm,NULL);
    OpenMX_ForceNonlocal(&atomic,expected_cdm,expected_edm,NULL,NULL);
    OpenMX_ForceCoreHole(&atomic,expected_cdm,expected_edm,NULL);
  }
  assert(stage_calls==12);
  printf("PASS: %d Force invocations with real topology/RAII owner, argument forwarding and 1-8 workers\n",invocations);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-force-lifetime-") as directory:
    tmp = Path(directory)
    for backend, name in (("cuda", "source/Force.c"), ("hip", "source/backends/hip/src/Force.c")):
        source = (ROOT / name).read_text()
        arguments = re.search(r"typedef struct \{\s*OpenMXHamiltonianGpuState \*hamiltonian_state;\s*OpenMXDensityGpuState \*density_state;\s*const OpenMXAtomicViews \*atomic;.*?\} ForceArguments;", source, re.S)
        assert arguments, name
        first = source.index("static double Force_Impl(")
        declaration = source[first:source.index(";", first)]
        adapter = function((ROOT / "source/openmx_solver_workspace.h").read_text(),
                           "static inline void OpenMX_Solver_Orbitals(")
        program = PREFIX + adapter + STAGES + arguments[0] + "\n" + declaration + ";\n"
        program += function(source, "static int Force_Operation(") + "\n" + function(source, "double Force(")
        program += "\n" + declaration + BODY + MAIN
        c = tmp / (backend + ".c"); c.write_text(program); exe = c.with_suffix("")
        command = shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "")) + [
            "-std=c11", "-O2", "-fopenmp", "-I" + str(ROOT / "core/include"), "-I" + str(ROOT / "source"), str(c), str(CORE),
            "-Wl,-rpath," + str(CORE.parent), "-lm", "-o", str(exe)]
        subprocess.run(command, check=True)
        subprocess.run([str(exe)], check=True, timeout=90, preexec_fn=bounded_stack)

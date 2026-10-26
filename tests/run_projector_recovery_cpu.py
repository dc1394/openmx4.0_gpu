#!/usr/bin/env python3
"""Check capacity packing against the established unsplit projector kernels."""
import ast
import os
import resource
from pathlib import Path
from task_worker_test_bridge import compile_task_test
import shlex
import subprocess
import tempfile
import force_gpu_fixture
from run_force4b_stream_cpu import extract_function, extract_struct
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include <cuda_runtime.h>','').replace('#include "set_cuda_default_device_from_local_rank.h"','')
s=(root/'source/Force.c').read_text()
s=s.replace('#include "openmx_force_gpu_resources.h"',(root/'source/openmx_force_gpu_resources.h').read_text())
def function(s,name):
 a=s.index(name);b=s.index('{',a)+1;depth=1
 while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[a:b]
def typedef(s,name):
 if name=='Force4BGpuItem':return extract_struct(s,name)
 if name=='ForceHNLGpuContext':
  header=(root/'source/openmx_force_hnl_gpu.h').read_text()
  a=header.index('struct ForceHNLGpuContext {');b=header.index('};',a)+2
  return 'typedef struct ForceHNLGpuContext ForceHNLGpuContext;\n'+header[a:b]
 b=s.index('} '+name+';')+len('} '+name+';');a=s.rfind('typedef struct {',0,b);return s[a:b]
body=body.replace('int openmx_gpu_task_execute(', 'int fixture_task_execute(')
functions='\ntypedef struct OpenMXNativeResources OpenMXNativeResources;\nstatic int mpi_comm_level1=0,fp32=0,peer_rounds,empty_dispatches;\n#define MPI_INT 0\n#define MPI_MAX 0\nstatic void MPI_Allreduce(const void *s,void *d,int n,int t,int op,int comm){if(peer_rounds){peer_rounds--;*(int*)d=1;}else memcpy(d,s,n*sizeof(int));}\nstatic int openmx_fp32_enabled(void){return fp32;}\nstatic void Force4B_gpu_abort(const char *msg){fprintf(stderr,"%s",msg);abort();}\n'
functions+='int openmx_gpu_task_execute(MPI_Comm c,OpenMXGpuTask *t,int e,const char *l){if(!t->count)empty_dispatches++;return fixture_task_execute(c,t,e,l); }\n'
for name in ('Force4BGpuItem','Force4BGpuContext','ForceHNLGpuContext'):functions+=typedef(s,name)+'\n'
functions+='#include "openmx_core_force_recovery.h"\ntypedef OpenMXForceHnlRecoveryItem ForceHNLCapacityItem;\n'
functions+='#define OpenMXGpuMemcpy cudaMemcpy\n#define OPENMX_GPU_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost\n#define OPENMX_GPU_SUCCESS cudaSuccess\n'
functions+=extract_struct(s,'Force4BCapacityParameters')
functions+=extract_function(s,'Force4B_capacity_kernel')+'\n'
functions+=extract_function(s,'Force_HNL_capacity_kernel')+'\n'
functions+=(root/'source/openmx_force_recovery.h').read_text()+'\n'
# Host SIMD policy of the recovery: the collective packet loop (legacy) and the
# rank-local SIMD bodies (widest host mode) must both match the owner kernels.
functions+='static int host_simd_policy=OPENMX_HOST_SIMD_LEGACY;\nint OpenMX_HostSimdMode(void){return host_simd_policy;}\n'
# These unchanged owner kernels are independent of capacity packing/sharding.
functions+=function(s,'static void Force4B_GpuCase1Kernel_Single(')+'\n'
# Skip the forward declaration of the second kernel.
a=s.index('/* Both full and one-centre batches');functions+=function(s[a:],'static void Force4B_GpuCase2Kernel_Single(')+'\n'
oracle=(root/'tests/fixtures/hnl_before_capacity.c').read_text()
functions+='static void reference1(int nitems_c,const Force4BGpuItem *items,const size_t *krowA,const size_t *krowB,const int *krow_halo,const int *krow_ene,const int *krow_nlp,const double *cdm_pref,const double *halo,const double *ene,const double *flat,size_t flat_stride,double *item_f){\n'+oracle.split('/* CASE 1 */\n')[1].split('/* CASE 2 */')[0]+'}\n'
functions+='static void reference2(int nitems_c,const Force4BGpuItem *items,const int *item_h0,const int *item_nlp,const int *item_ene,const size_t *item_a,const size_t *item_b,const double *cdm_pref,const double *c2,const double *ene,const double *flat,size_t c2_stride,double *item_f){\n'+oracle.split('/* CASE 2 */\n')[1]+'}\n'
main=r'''
int main(int argc,char **argv){
 int N=argc>1?atoi(argv[1]):17,STRIDE=32*N,MAXK=N*3;
 Force4BGpuItem items[N];size_t a[MAXK],b[MAXK];int halo[MAXK],ene[MAXK],nlp[MAXK],h0[N];
 double flat[4*STRIDE],halodata[STRIDE],c2[4*STRIDE],energies[32],weights[N*16],out[3*N+2],expected[3*N];
 float flatf[4*STRIDE],halof[STRIDE],c2f[4*STRIDE];
 for(int i=0;i<4*STRIDE;i++){flat[i]=sin(.03*i);c2[i]=cos(.05*i);flatf[i]=flat[i];c2f[i]=c2[i];}
 for(int i=0;i<STRIDE;i++){halodata[i]=sin(.07*i+.5);halof[i]=halodata[i];}
 for(int i=0;i<32;i++)energies[i]=.03*(i-4);
 for(int i=0;i<N*16;i++)weights[i]=sin(.09*i);
 for(int i=0;i<MAXK;i++){a[i]=3*i;b[i]=2*i+7;halo[i]=i%2;ene[i]=i%7;nlp[i]=i%4+1;}
 int cases=0;
 for(int policy=0;policy<2;policy++){
 host_simd_policy=policy?openmx_core_host_simd_mode(NULL):OPENMX_HOST_SIMD_LEGACY;
 /* the rank-local bodies ignore the worker tiers: run them once */
 const int worker_max=policy?1:4,tier_max=policy?1:4;
 for(int kind=1;kind<=2;kind++){
  for(int i=0;i<N;i++){items[i]=(Force4BGpuItem){(size_t)i*16,i*3,i%4,2+i%3,1+i%4};h0[i]=i%3;}
  ForceHNLGpuContext g={0};g.flat=flat;g.halo=halodata;g.c2=c2;g.ene=energies;g.flat_stride=g.c2_stride=STRIDE;
  if(kind==1)reference1(N,items,a,b,halo,ene,nlp,weights,halodata,energies,flat,STRIDE,expected);
  else reference2(N,items,h0,nlp,ene,a,b,weights,c2,energies,flat,STRIDE,expected);
  for(workers=1;workers<=worker_max;workers++)for(int tier=0;tier<tier_max;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
   for(int i=0;i<3*N+2;i++)out[i]=1234;
   Force_HNL_capacity(kind,&g,N,items,a,b,halo,ene,nlp,h0,weights,out+1,0);
   for(int i=0;i<3*N;i++)assert(fabs(out[i+1]-expected[i])<1e-11*(1+fabs(expected[i])));
   assert(out[0]==1234 && out[3*N+1]==1234);cases++;
  }
  for(fp32=0;fp32<2;fp32++){
   Force4BGpuContext g4={0};g4.num_proj=5;g4.flat_host=flatf;g4.halo_host=halof;g4.c2_host=c2f;g4.flat_stride=g4.c2_stride=STRIDE;
   if(kind==1)Force4B_GpuCase1Kernel_Single(5,flatf,halof,STRIDE,N,items,a,b,halo,weights,expected);
   else Force4B_GpuCase2Kernel_Single(5,c2f,STRIDE,N,items,a,b,weights,expected);
   for(workers=1;workers<=worker_max;workers++)for(int tier=0;tier<tier_max;tier++){
    local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
    for(int i=0;i<3*N+2;i++)out[i]=1234;
    Force4B_capacity(kind,&g4,N,items,a,b,halo,weights,out+1,0);
    for(int i=0;i<3*N;i++)assert(fabs(out[i+1]-expected[i])<1e-11*(1+fabs(expected[i])));
    assert(out[0]==1234 && out[3*N+1]==1234);cases++;
   }
  }
 }
 /* Done/empty ranks join the collective loop under the legacy policy; the
    host SIMD policy has no collective step, no dispatch and no output. */
 for(int kind=1;kind<=2;kind++)for(int done=0;done<2;done++){
  Force4BGpuContext g4={0};g4.num_proj=5;ForceHNLGpuContext g={0};
  for(int i=0;i<3*N+2;i++)out[i]=4321;
  peer_rounds=2;empty_dispatches=0;
  Force4B_capacity(kind,&g4,done?N:0,items,a,b,halo,weights,out+1,done);
  if(policy)assert(peer_rounds==2 && !empty_dispatches);else assert(!peer_rounds && empty_dispatches==2);
  peer_rounds=2;empty_dispatches=0;
  Force_HNL_capacity(kind,&g,done?N:0,items,a,b,halo,ene,nlp,h0,weights,out+1,done);
  if(policy)assert(peer_rounds==2 && !empty_dispatches);else assert(!peer_rounds && empty_dispatches==2);
  for(int i=0;i<3*N+2;i++)assert(out[i]==4321);
 }
 }
 peer_rounds=0;
 printf("PASS: %d HNL/VNA force recovery cases against owner kernels (ragged pack, halo, FP32/FP64, local/remote/CPU, empty/done ranks; host SIMD policies legacy and %s)\n",cases,openmx_core_host_simd_name(openmx_core_host_simd_mode(NULL)));
}
'''
def bounded_stack():
 _,hard=resource.getrlimit(resource.RLIMIT_STACK)
 resource.setrlimit(resource.RLIMIT_STACK,(8*1024*1024,hard))

with tempfile.TemporaryDirectory(prefix='openmx-projector-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/"cuda_runtime.h").write_text("/* Runtime mock is provided by the fixture prefix. */\n")
 (p/"openmx_gpu_device.h").write_text("/* Rank availability mock is provided by the fixture prefix. */\n")
 (p/'check.c').write_text(prefix+body+functions+main)
 compile_task_test(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-pie','-no-pie','-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),*force_gpu_fixture.core_library_flags(),'-lm','-o',str(p/'check')],check=True)
 for count in (17,513):
  subprocess.run([str(p/'check'),str(count)],check=True,timeout=60,preexec_fn=bounded_stack)

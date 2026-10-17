#!/usr/bin/env python3
"""Check capacity packing against the established unsplit projector kernels."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
s=(root/'source/Force.c').read_text()
def function(s,name):
 a=s.index(name);b=s.index('{',a)+1;depth=1
 while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[a:b]
def typedef(s,name):
 b=s.index('} '+name+';')+len('} '+name+';');a=s.rfind('typedef struct {',0,b);return s[a:b]
functions='\nstatic int mpi_comm_level1=0,fp32=0;\n#define MPI_INT 0\n#define MPI_MAX 0\nstatic void MPI_Allreduce(const void *s,void *d,int n,int t,int op,int comm){memcpy(d,s,n*sizeof(int));}\nstatic int openmx_fp32_enabled(void){return fp32;}\nstatic void Force4B_gpu_abort(const char *msg){fprintf(stderr,"%s",msg);abort();}\n'
for name in ('Force4BGpuItem','Force4BGpuContext','ForceHNLGpuContext'):functions+=typedef(s,name)+'\n'
a=s.index('typedef struct {int kind,ian,jan,kn,h0,nlp;}');b=s.index('/* the unified first-case batch:',a);functions+=s[a:b]
a=s.index('typedef struct {int kind,num_proj,fp32;}');b=s.index('static void Force4B_GpuCase1Run(',a);functions+=s[a:b]
# These unchanged owner kernels are independent of capacity packing/sharding.
functions+=function(s,'static void Force4B_GpuCase1Kernel_Single(')+'\n'
# Skip the forward declaration of the second kernel.
a=s.index('/* Both full and one-centre batches');functions+=function(s[a:],'static void Force4B_GpuCase2Kernel_Single(')+'\n'
oracle=(root/'tests/fixtures/hnl_before_capacity.c').read_text()
functions+='static void reference1(int nitems_c,const Force4BGpuItem *items,const size_t *krowA,const size_t *krowB,const int *krow_halo,const int *krow_ene,const int *krow_nlp,const double *cdm_pref,const double *halo,const double *ene,const double *flat,size_t flat_stride,double *item_f){\n'+oracle.split('/* CASE 1 */\n')[1].split('/* CASE 2 */')[0]+'}\n'
functions+='static void reference2(int nitems_c,const Force4BGpuItem *items,const int *item_h0,const int *item_nlp,const int *item_ene,const size_t *item_a,const size_t *item_b,const double *cdm_pref,const double *c2,const double *ene,const double *flat,size_t c2_stride,double *item_f){\n'+oracle.split('/* CASE 2 */\n')[1]+'}\n'
main=r'''
int main(void){
 enum{N=17,STRIDE=512,MAXK=N*3};
 Force4BGpuItem items[N];size_t a[MAXK],b[MAXK];int halo[MAXK],ene[MAXK],nlp[MAXK],h0[N];
 double flat[4*STRIDE],halodata[STRIDE],c2[4*STRIDE],energies[32],weights[N*16],out[3*N+2],expected[3*N];
 float flatf[4*STRIDE],halof[STRIDE],c2f[4*STRIDE];
 for(int i=0;i<4*STRIDE;i++){flat[i]=sin(.03*i);c2[i]=cos(.05*i);flatf[i]=flat[i];c2f[i]=c2[i];}
 for(int i=0;i<STRIDE;i++){halodata[i]=sin(.07*i+.5);halof[i]=halodata[i];}
 for(int i=0;i<32;i++)energies[i]=.03*(i-4);
 for(int i=0;i<N*16;i++)weights[i]=sin(.09*i);
 for(int i=0;i<MAXK;i++){a[i]=3*i;b[i]=2*i+7;halo[i]=i%2;ene[i]=i%7;nlp[i]=i%4+1;}
 int cases=0;
 for(int kind=1;kind<=2;kind++){
  for(int i=0;i<N;i++){items[i]=(Force4BGpuItem){(size_t)i*16,i*3,i%4,2+i%3,1+i%4};h0[i]=i%3;}
  ForceHNLGpuContext g={0};g.flat=flat;g.halo=halodata;g.c2=c2;g.ene=energies;g.flat_stride=g.c2_stride=STRIDE;
  if(kind==1)reference1(N,items,a,b,halo,ene,nlp,weights,halodata,energies,flat,STRIDE,expected);
  else reference2(N,items,h0,nlp,ene,a,b,weights,c2,energies,flat,STRIDE,expected);
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
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
   for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
    local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
    for(int i=0;i<3*N+2;i++)out[i]=1234;
    Force4B_capacity(kind,&g4,N,items,a,b,halo,weights,out+1,0);
    for(int i=0;i<3*N;i++)assert(fabs(out[i+1]-expected[i])<1e-11*(1+fabs(expected[i])));
    assert(out[0]==1234 && out[3*N+1]==1234);cases++;
   }
  }
 }
 printf("PASS: %d HNL/VNA force recovery cases against owner kernels (ragged pack, halo, FP32/FP64, local/remote/CPU)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-projector-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/'check.c').write_text(prefix+body+functions+main)
 subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),'-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)

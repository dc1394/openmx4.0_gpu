#!/usr/bin/env python3
"""Sparse real/complex DM recovery packets, phases, padding, and CPU fallback."""
import ast,os,shlex,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=''
for name in ('openmx_gpu_task.c','openmx_gpu_dm_collective.c'):
 body+=(root/'source'/name).read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','').replace('#include "openmx_mgpu_runtime.h"','')+'\n'
prefix=prefix.replace('static int gpu_rank_device_usable',
 'static int mock_multi_node=0,last_pool_kind=-1;\nstatic int gpu_rank_device_usable')
prefix=prefix.replace('(void)kind;p->items=n;', 'last_pool_kind=kind;p->items=n;')
prefix+='\nstatic int openmx_mgpu_multi_node(void){return mock_multi_node;}\n'
stub=r'''
int openmx_gpu_dm_try_entries(int kind,int n,int states,int stride,int entries,const void *vectors,int host,
 const int *a,const int *b,const int *ph,int pc,const double *pr,const double *pi,const double *w,const double *ew,const double *pw,
 double *const output[10],int acc){return 0;}
'''
main=r'''
#include <complex.h>
int main(void){
 enum{N=7,K=17,LD=23,E=19,P=5,NN=14*23};
 double real[NN],occ[K],energy[K],partial[K],pr[P],pi[P],output[10][E+2],expected[10][E];
 OpenMXDMComplex vec[NN];int a[E],b[E],ph[E],cases=0;double *outputs[10];
 for(int i=0;i<NN;i++){real[i]=sin(.2*i);vec[i]=(OpenMXDMComplex){real[i],cos(.7*i)};}
 for(int k=0;k<K;k++){occ[k]=.3+.02*k;energy[k]=-.4+.03*k;partial[k]=.1+.01*k;}
 for(int i=0;i<P;i++){pr[i]=cos(.3*i);pi[i]=sin(.3*i);}
 for(int i=0;i<E;i++){a[i]=i%N;b[i]=(i*3+1)%N;ph[i]=i%P;}
 for(int c=0;c<10;c++)outputs[c]=output[c]+1;
 for(int kind=0;kind<=OPENMX_GPU_EDM_GATHER;kind++)for(int energy_on=0;energy_on<2;energy_on++)
 for(int phased=0;phased<2;phased++)for(int accumulate=0;accumulate<2;accumulate++)for(int host=0;host<2;host++){
  for(int p=0;p<E;p++){
   long double r[10]={0};long double complex dm[3]={0},edm[3]={0};
   if(kind>=OPENMX_GPU_DM_GATHER){
    OpenMXDMComplex x=vec[a[p]+b[p]*2*N],y=vec[a[p]+N+(b[p]+N)*2*N],z=vec[a[p]+(b[p]+N)*2*N];
    r[0]=x.x;r[1]=y.x;r[2]=z.x;r[3]=z.y;if(kind==OPENMX_GPU_DM_GATHER){r[4]=x.y;r[5]=y.y;}
   }else for(int k=0;k<K;k++){
    if(kind==OPENMX_GPU_DM_REAL){
     long double product=(long double)real[a[p]*LD+k]*real[b[p]*LD+k];
     r[0]+=occ[k]*product;if(energy_on)r[1]+=occ[k]*energy[k]*product;r[2]+=partial[k]*product;
    }else{
     OpenMXDMComplex av=vec[a[p]*LD+k],bv=vec[b[p]*LD+k],cv=vec[(a[p]+N)*LD+k],dv=vec[(b[p]+N)*LD+k];
     long double complex va=av.x+I*(long double)av.y,vb=bv.x+I*(long double)bv.y,vc=cv.x+I*(long double)cv.y,vd=dv.x+I*(long double)dv.y;
     long double complex prod[3]={conjl(va)*vb,conjl(vc)*vd,conjl(va)*vd};
     for(int c=0;c<3;c++){
      dm[c]+=prod[c]*occ[k]*(kind==OPENMX_GPU_DM_COL?(long double)occ[k]:1);
      if(energy_on)edm[c]+=prod[c]*energy[k]*(kind==OPENMX_GPU_DM_COL?(long double)occ[k]*occ[k]:1);
     }
    }
   }
   if(kind==OPENMX_GPU_DM_COL || kind==OPENMX_GPU_DM_NC){
    long double complex phase=phased?pr[ph[p]]+I*(long double)pi[ph[p]]:1;
    for(int c=0;c<3;c++){dm[c]*=phase;edm[c]*=phase;}
    if(kind==OPENMX_GPU_DM_COL){r[0]=creall(dm[0]);r[1]=creall(edm[0]);}
    else{r[0]=creall(dm[0]);r[1]=creall(dm[1]);r[2]=creall(dm[2]);r[3]=cimagl(dm[2]);r[4]=cimagl(dm[0]);r[5]=cimagl(dm[1]);r[6]=creall(edm[0]);r[7]=creall(edm[1]);r[8]=creall(edm[2]);r[9]=cimagl(edm[2]);}
   }
   for(int c=0;c<10;c++)expected[c][p]=(accumulate?7:0)+(double)r[c];
  }
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
   for(int c=0;c<10;c++)for(int i=0;i<E+2;i++)output[c][i]=7;
   openmx_gpu_dm_collective(0,kind,N,K,LD,E,kind==OPENMX_GPU_DM_REAL?(void*)real:(void*)vec,host,a,b,phased?ph:NULL,P,pr,pi,occ,energy_on?energy:NULL,kind==OPENMX_GPU_DM_REAL?partial:NULL,outputs,accumulate);
   for(int c=0;c<10;c++){
    assert(output[c][0]==7 && output[c][E+1]==7);
    for(int i=0;i<E;i++)assert(isfinite(outputs[c][i]) && fabs(outputs[c][i]-expected[c][i])<1e-12*(1+fabs(expected[c][i])));
   }
   cases++;
  }
 }
 openmx_gpu_dm_collective(0,0,0,0,0,0,NULL,0,NULL,NULL,NULL,0,NULL,NULL,NULL,NULL,NULL,NULL,0);
 local_fail=remote_fail=fail_alloc=0;workers=2;
 for(mock_multi_node=0;mock_multi_node<=1;mock_multi_node++)for(int phased=0;phased<=1;phased++){
  last_pool_kind=-1;
  openmx_gpu_dm_collective(0,OPENMX_GPU_DM_COL,N,K,LD,E,vec,1,a,b,phased?ph:NULL,P,pr,pi,occ,energy,NULL,outputs,0);
  assert(last_pool_kind==(phased && !mock_multi_node?2:1));
 }
 printf("PASS: %d sparse DM recovery cases, long-double oracle and empty helpers\n",cases+1);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-dm-capacity-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('''#pragma once
typedef int MPI_Comm;
#define MPI_COMM_NULL (-1)
#define MPI_IN_PLACE ((void*)1)
#define MPI_INT 1
#define MPI_MAX 2
static int MPI_Allreduce(void *s,void *r,int n,int type,int op,MPI_Comm comm){return 0;}
static int MPI_Abort(MPI_Comm c,int err){abort();}
''')
 (p/'check.c').write_text(prefix+stub+body+main)
 subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g','-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),'-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)

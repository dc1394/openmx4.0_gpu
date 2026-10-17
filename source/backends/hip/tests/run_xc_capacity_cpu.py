#!/usr/bin/env python3
"""Production XC halo packets/rotation/derivatives under ASan and UBSan.

OpenMP loops execute on the host. The complete kernel and pack/run/commit
functions are extracted unchanged; this checks indexing and transaction
lifetime, while GPU integration tests check device arithmetic/binding.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
source=(root/"source/Set_XC_Grid.c").read_text()
kernel="#define SXG_DEN_MIN 1.0e-14\n"+source[source.index("#pragma omp declare target"):source.index("static int Set_XC_Grid_GPU(")]
prefix=r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_workers.h"
#define PI 3.14159265358979323846264338327950288
#define hipSuccess 0
static int workers=1,allocation_fail=0,fail_once=0,copy_fail=0;
static int hipMalloc(void **p,size_t n){if(fail_once){fail_once--;*p=NULL;return 1;}*p=allocation_fail?NULL:malloc(n);return !*p;}
static int hipFree(void *p){free(p);return 0;}
static int hipGetLastError(void){return 0;}
static int hipDeviceSynchronize(void){return 0;}
static void openmx_target_clear_freelists(void){}
static int gpu_rank_device_usable(void){return 1;}
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
static int hipMemcpy(void *d,const void *s,size_t n,int mode){if(copy_fail==mode)return 1;memcpy(d,s,n);return 0;}
static int openmx_target_upload(void *d,const void *s,size_t n){memcpy(d,s,n);return 1;}
static int openmx_target_download(void *d,const void *s,size_t n){memcpy(d,s,n);return 1;}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers *p,size_t n,int kind,OpenMXGpuWorkerCost cost,void *ctx){
 (void)kind;p->items=n;p->count=n<(size_t)workers?(int)n:workers;
 for(;p->count>0;p->count--){
  int fit=1;for(int i=0;i<p->count;i++){p->devices[i]=i;size_t nb=cost(ctx,i,p->count);assert(nb>0);if(nb==SIZE_MAX)fit=0;}
  if(fit)return 1;
 }
 return 0;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers *p,int (*f)(void*,int,int),void *ctx,const char *name){
 (void)name;int ok=1,started[64]={0};
 for(int phase=0;phase<4;phase++){
  if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
  for(int i=0;i<p->count;i++){
   if(phase==OPENMX_GPU_WORKER_RELEASE && !started[i])continue;
   if(phase==OPENMX_GPU_WORKER_PREPARE)started[i]=1;
   if(!f(ctx,i,phase))ok=0;
  }
 }
 return ok;
}
'''
main=r'''
enum {N1=11,N2=5,N3=6,N=N1*N2*N3};
static double input[6][N],output[16][N+2],reference[16][N+2],inv[729];
static void setup(SXG_Work *w,int mode,int rotate,int pcc,int derivatives){
 memset(w,0,sizeof(*w));w->XC_P_switch=mode;w->rotate=rotate;w->pcc=pcc;
 w->Ng1=N1;w->Ng2=N2;w->Ng3=N3;w->last=N1;w->inv=inv;
 for(int f=0;f<6;f++)for(int i=0;i<N;i++)input[f][i]=f<2?0.05+0.003*sin(i*0.01+f):0.01+0.001*cos(i*0.02+f);
 for(int f=0;f<16;f++)for(int i=0;i<N+2;i++)output[f][i]=12345;
 for(int f=0;f<4;f++){w->den[f]=input[f];w->vxc[f]=output[f]+1;}
 w->pcc_den[0]=input[4];w->pcc_den[1]=input[5];
 static double *dd[6],*dx[6],**ddp[2]={dd,dd+3},**dxp[2]={dx,dx+3};
 for(int i=0;i<6;i++){dd[i]=output[4+i]+1;dx[i]=output[10+i]+1;}
 if(derivatives){w->dden=ddp;w->dexc=dxp;}
}
static void check(void){
 for(int f=0;f<16;f++)for(int i=0;i<N+2;i++){
  assert(isfinite(output[f][i]));
  if(fabs(output[f][i]-reference[f][i])>1e-12){fprintf(stderr,"mismatch %d %d %.17g %.17g\n",f,i,output[f][i],reference[f][i]);abort();}
 }
}
int main(void){
 /* A finite dense interpolation matrix also exercises remote payload
    transfer instead of relying on identical rank-local constants. */
 for(int i=0;i<729;i++)inv[i]=0.001*sin(i+1);
 int cases=0;
 for(int mode=0;mode<3;mode++)for(int rot=0;rot<2;rot++)for(int pcc=0;pcc<2;pcc++)for(int deriv=0;deriv<2;deriv++){
  SXG_Work w;setup(&w,mode,rot,pcc,deriv);workers=1;assert(SXG_execute(&w));memcpy(reference,output,sizeof(output));
  for(workers=2;workers<=4;workers++){
   setup(&w,mode,rot,pcc,deriv);assert(SXG_execute(&w));check();cases++;
   setup(&w,mode,rot,pcc,deriv);
   void *packets[3]={0},*results[3]={0};size_t bytes[3],rb[3];
   for(int i=0;i<3;i++){
    size_t lo=openmx_gpu_worker_cut(N1,i,3),hi=openmx_gpu_worker_cut(N1,i+1,3);
    assert(SXG_pack(&w,lo,hi,&packets[i],&bytes[i]));
    assert(SXG_remote(NULL,packets[i],bytes[i],&results[i],&rb[i]));
   }
   for(int i=0;i<3;i++){
    SXG_commit(&w,openmx_gpu_worker_cut(N1,i,3),openmx_gpu_worker_cut(N1,i+1,3),results[i],rb[i]);
    free(packets[i]);free(results[i]);
   }
   check();cases++;
  }
  setup(&w,mode,rot,pcc,deriv);allocation_fail=1;assert(!SXG_execute(&w));allocation_fail=0;
  for(int f=0;f<16;f++)for(int i=0;i<N+2;i++)assert(output[f][i]==12345);
  for(copy_fail=1;copy_fail<=2;copy_fail++) {
   setup(&w,mode,rot,pcc,deriv);assert(!SXG_execute(&w));
   for(int f=0;f<16;f++)for(int i=0;i<N+2;i++)assert(output[f][i]==12345);
  }
  copy_fail=0;
  setup(&w,mode,rot,pcc,deriv);workers=4;fail_once=1;
  assert(SXG_execute(&w));assert(fail_once==0);check();
 }
 printf("PASS: %d XC local/remote slab comparisons plus allocation rollback and guards\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-xc-capacity-") as folder:
    p=Path(folder);(p/"check.c").write_text(prefix+kernel+main)
    subprocess.run(shlex.split(os.environ.get("CC","cc"))+["-std=c11","-O1","-g","-Wall","-Wextra",
        "-Wno-unknown-pragmas","-Wno-unused-function","-fsanitize=address,undefined","-fno-sanitize-recover=all",
        "-I"+str(root/"source"),str(p/"check.c"),"-lm","-o",str(p/"check")],check=True)
    subprocess.run([str(p/"check")],check=True,timeout=45)

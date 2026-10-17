#!/usr/bin/env python3
"""Compare VNA capacity shards with frozen unsplit projector kernels."""
import ast
import os
import re
from pathlib import Path
import shlex
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
s=(root/'source/Set_ProExpn_VNA.c').read_text()
def function(s,name):
 a=s.index(name);b=s.index('{',a)+1;depth=1
 while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
 return s[a:b]
functions='#define SETPRO_BES_LMAX 30\n'+function(s,'static void SetPro_Spherical_Bessel2_dev(')+'\n'
old=(root/'tests/fixtures/setpro_before_capacity.c').read_text()
for tag in ('DSVNA','VNA23'):
 a=s.index('typedef struct {int mesh,'+('num_proj' if tag=='DSVNA' else 'l0max'))
 b=s.index('static void SetPro_'+tag+'_GpuRun_Part(',a)
 code=s[a:b];functions+=code
 kernel=function(code,'static void SetPro_'+tag+'_capacity_kernel(')
 head=kernel[:kernel.index('#pragma omp target teams distribute parallel for')]
 oracle=old.split('/* '+tag+' */\n')[1]
 if tag=='DSVNA':oracle=oracle.split('/* VNA23 */')[0]
 lines=[];continuation=False
 pointers=','.join(re.findall(r'(?:double|float|int) \*(\w+)=v\[',head))
 for line in oracle.splitlines():
  if continuation:
   continuation=line.rstrip().endswith(chr(92));continue
  if line.lstrip().startswith('#pragma'):
   continuation=line.rstrip().endswith(chr(92))
   if os.environ.get('OPENMX_TEST_GPU')=='1':
    if 'parallel loop' in line:
     lines.append(line.rstrip().rstrip(chr(92))+' deviceptr('+pointers+') if(gpu)')
    else:lines.append(line)
   continue
  lines.append(line)
 oracle='\n'.join(lines)+'\n'
 functions+=head.replace('SetPro_'+tag+'_capacity_kernel','reference_'+tag)+oracle.replace('GL_Mesh','mesh')+'}\n'
main=r'''
static void fill(double *x,size_t n,double base){for(size_t i=0;i<n;i++)x[i]=base+.003*(i%17);}
int main(void){
 enum{N=17,SP=2,MESH=6,CMB=2,LL=3,NM=3,MAXNO=4,MN=16};
 double pro00[SP*CMB*MESH],pro01[SP*CMB*MESH],vnab[SP*2*MESH],gnt[2*3*2*3*3*5],c2r[20]={0},norm[MESH],weight[MESH],crude[SP*MESH],prod[SP*CMB*CMB*LL*MESH],gnt2[2*3*2*3*3];
 int c2ro[3]={0,1,10},lfi[SP]={2,2},ncmb[SP]={2,1},L0[SP*CMB]={0,1,0,0},num0[SP*CMB]={0,1,0,0},L1[2]={0,1},num1[2]={0,1};
 double r[N],siT[N],coT[N],siP[N],coP[N],sh[N*9*6];int spe[N],hwan[N],h0[N],tno[N];
 fill(pro00,sizeof(pro00)/8,.2);fill(pro01,sizeof(pro01)/8,.1);fill(vnab,sizeof(vnab)/8,.3);fill(gnt,sizeof(gnt)/8,.04);fill(crude,sizeof(crude)/8,.11);fill(prod,sizeof(prod)/8,.12);fill(gnt2,sizeof(gnt2)/8,.06);fill(sh,sizeof(sh)/8,.3);
 c2r[0]=1;for(int i=0;i<3;i++)c2r[2*(1+i*3+i)]=1;
 for(int i=0;i<MESH;i++){norm[i]=.17*i;weight[i]=.15+i*.01;}
 for(int i=0;i<N;i++){r[i]=i*.2;siT[i]=sin(.3+i*.1);coT[i]=cos(.3+i*.1);siP[i]=sin(.7+i*.1);coP[i]=cos(.7+i*.1);spe[i]=i%2;hwan[i]=(i+1)%2;h0[i]=i%4==0;tno[i]=spe[i]?1:4;}
 int cases=0;
 for(int type=0;type<2;type++){
  SetProDSVNACapacityParameters dsvna={MESH,4,2,2,2,4,1,1,MN};
  SetProVNA23CapacityParameters vna={MESH,1,2,2,MN,MAXNO,1.7};
  OpenMXGpuTask task={0};task.count=N;task.outputs=1;task.scratch=5;
  task.kernel=type?SetPro_VNA23_capacity_kernel:SetPro_DSVNA_capacity_kernel;
  task.parameters=type?(const void*)&vna:(const void*)&dsvna;task.parameter_bytes=type?sizeof(vna):sizeof(dsvna);
  int v=0;
#define TABLE(x) task.view[v++]=(OpenMXGpuTaskView){x,sizeof(x),0}
#define ITEM(x) task.view[v++]=(OpenMXGpuTaskView){x,0,sizeof(x[0])}
  if(!type){TABLE(pro00);TABLE(pro01);TABLE(vnab);TABLE(gnt);TABLE(c2r);TABLE(c2ro);TABLE(lfi);TABLE(ncmb);TABLE(L0);TABLE(num0);TABLE(L1);TABLE(num1);TABLE(norm);}
  else{TABLE(crude);TABLE(prod);TABLE(gnt2);TABLE(c2r);TABLE(c2ro);TABLE(lfi);TABLE(ncmb);TABLE(L0);TABLE(num0);TABLE(norm);TABLE(weight);}
  ITEM(r);ITEM(siT);ITEM(coT);ITEM(siP);ITEM(coP);ITEM(spe);ITEM(hwan);ITEM(h0);if(!type){ITEM(tno);}
  task.view[v++]=(OpenMXGpuTaskView){sh,0,9*6*8};task.inputs=v;
  size_t width=type?8:4,out_stride=4*MN*width;
  unsigned char output[N*4*MN*8+16],reference[N*4*MN*8],cpu_reference[N*4*MN*8];task.view[v++]=(OpenMXGpuTaskView){output+8,0,out_stride};
  task.view[v++]=(OpenMXGpuTaskView){NULL,0,LL*MESH*8};task.view[v++]=(OpenMXGpuTaskView){NULL,0,LL*MESH*8};
  task.view[v++]=(OpenMXGpuTaskView){NULL,0,LL*CMB*2*8};task.view[v++]=(OpenMXGpuTaskView){NULL,0,LL*CMB*2*8};
  size_t elements=CMB*NM*2*NM;task.view[v++]=(OpenMXGpuTaskView){NULL,0,elements*8*8};
  void *oracle[OPENMX_GPU_TASK_VIEWS];
  for(int i=0;i<task.inputs;i++){
   size_t bytes=task.view[i].bytes?task.view[i].bytes:N*task.view[i].item_bytes;
   oracle[i]=malloc(bytes);memcpy(oracle[i],task.view[i].data,bytes);
  }
  memset(reference,0,sizeof(reference));oracle[task.inputs]=calloc(1,sizeof(reference));
  for(int i=task.inputs+1;i<v;i++)oracle[i]=calloc(N,task.view[i].item_bytes);
  int ref_gpu=getenv("OPENMX_TEST_GPU")!=NULL;
  if(type)reference_VNA23(0,N,&vna,oracle,ref_gpu);else reference_DSVNA(0,N,&dsvna,oracle,ref_gpu);
  memcpy(reference,oracle[task.inputs],sizeof(reference));
  memset(oracle[task.inputs],0,sizeof(reference));
  if(type)reference_VNA23(0,N,&vna,oracle,0);else reference_DSVNA(0,N,&dsvna,oracle,0);
  memcpy(cpu_reference,oracle[task.inputs],sizeof(cpu_reference));
  for(int i=0;i<v;i++)free(oracle[i]);
  for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
   local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;memset(output,0x55,sizeof(output));
   openmx_gpu_task_execute(0,&task,1,"projector");
   for(int i=0;i<N*4*MN;i++){
    double got=type?((double*)(output+8))[i]:((float*)(output+8))[i];const unsigned char *ref=tier>=2?cpu_reference:reference;double expected=type?((const double*)ref)[i]:((const float*)ref)[i];
    if(!(isfinite(got) && fabs(got-expected)<(type?1e-10:1e-6)*(1+fabs(expected)))){fprintf(stderr,"type=%d workers=%d tier=%d index=%d got=%.17g expected=%.17g\n",type,workers,tier,i,got,expected);abort();}
   }
   for(int i=0;i<8;i++){assert(output[i]==0x55);assert(output[8+N*out_stride+i]==0x55);}cases++;
  }
 }
 printf("PASS: %d VNA projector cases against original kernels (local/remote/CPU/padding)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-setpro-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/'check.c').write_text(prefix+body+functions+main)
 flags=['-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all']
 subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g']+flags+['-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),'-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)

#!/usr/bin/env python3
"""Capacity kernels: grid-list search, VNA contraction and radial integrals."""
import ast,os,shlex,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'tests/run_gpu_vector_cpu.py').read_text())
prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prefix' for t in n.targets))
body=(root/'source/openmx_gpu_task.c').read_text().replace('#include "hip_runtime_compat.h"','').replace('#include "set_hip_default_device_from_local_rank.h"','')
functions=''
for path,start,end in [('truncation.c','typedef struct {int grid,cell,l1,l2,l3;}','static int UCellBox_OLG_GPU('),('Set_ProExpn_VNA.c','typedef struct {int num_proj,num_rvna,fp32;}','static void SetProCapacity_HVNA_archive_row('),('Set_Nonlocal.c','static void SetNL_capacity_bessel(','/*\n   Two-phase evaluation'),('Set_OLP_Kin.c','typedef struct {int grid_dim,grid_last,combo_count,h_AN;double h;}','static void OLPKin_Radial_Distribute(')]:
 s=(root/'source'/path).read_text();a=s.index(start);b=s.index(end,a);functions+=s[a:b]
functions='#define SETNL_SB_ARR 40\n'+functions
main=r'''
static int cases;
static void check(OpenMXGpuTask *t,const double *ref,int outputs){
 for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
  local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;
  openmx_gpu_task_execute(0,t,1,"grid-test");
  for(int c=0;c<outputs;c++)for(size_t r=0;r<t->count;r++){
   double x=((double*)t->view[t->inputs+c].data)[r],y=ref[c*t->count+r];
   if(!(isfinite(x) && fabs(x-y)<2e-11*(1+fabs(y)))){fprintf(stderr,"workers=%d tier=%d c=%d row=%zu got=%.17g expected=%.17g\n",workers,tier,c,r,x,y);abort();}
  }cases++;
 }
}
int main(void){
 enum{N=19};
 {int grid[9]={2,2,2,5,5,8,9,9,12},cell[9]={0,1,2,0,2,1,0,1,2},atv[9]={0,0,0,1,0,0,0,1,0},ratv[27];
  for(int i=0;i<27;i++)ratv[i]=-1;ratv[13]=0;ratv[22]=1;ratv[16]=2;
  OLGCapacityQuery q[N];int output[N+2],expected[N];
  for(int i=0;i<N;i++)q[i]=(OLGCapacityQuery){i%14,i%3,i%5-2,(i/4)%3-1,0};
  OLGCapacityParameters p={9,1};OpenMXGpuTask t={0};t.count=N;t.inputs=5;t.outputs=1;t.kernel=OLG_capacity_kernel;t.parameters=&p;t.parameter_bytes=sizeof(p);
  t.view[0]=(OpenMXGpuTaskView){grid,sizeof(grid),0};t.view[1]=(OpenMXGpuTaskView){cell,sizeof(cell),0};t.view[2]=(OpenMXGpuTaskView){atv,sizeof(atv),0};t.view[3]=(OpenMXGpuTaskView){ratv,sizeof(ratv),0};
  t.view[4]=(OpenMXGpuTaskView){q,0,sizeof(q[0])};t.view[5]=(OpenMXGpuTaskView){output+1,0,sizeof(int)};
  for(int empty=0;empty<2;empty++){
   if(empty)p.nc=0;
   for(int i=0;i<N;i++){
    expected[i]=-1;int x=q[i].l1+atv[3*q[i].cell],y=q[i].l2+atv[3*q[i].cell+1],z=q[i].l3;
    if(abs(x)<=1 && abs(y)<=1 && abs(z)<=1)for(int j=0;j<p.nc;j++)
     if(grid[j]==q[i].grid && cell[j]==ratv[((x+1)*3+y+1)*3+z+1]){expected[i]=j;break;}
   }
   for(workers=1;workers<=4;workers++)for(int tier=0;tier<4;tier++){
    local_fail=tier>0;remote_fail=tier==2;fail_alloc=tier==3;for(int i=0;i<N+2;i++)output[i]=1234;
    openmx_gpu_task_execute(0,&t,1,"overlap");for(int i=0;i<N;i++)assert(output[i+1]==expected[i]);assert(output[0]==1234 && output[N+1]==1234);cases++;
   }
  }
 }
 for(int large=0;large<2;large++){
  int np=large?972:4,nr=large?108:2;
  size_t off[N+1],spoff[N+1];off[0]=spoff[0]=0;
  float a[N*4*np],b[N*4*np];int species[N*4],length[nr];double energy[2*nr],damping[N],out[N],expected[N];
  for(int r=0;r<nr;r++){length[r]=large?2*(r/12)+1:2*r+1;energy[r]=.4-.03*r;energy[nr+r]=-.2+.05*r;}
  for(int i=0;i<N*4*np;i++){a[i]=sin(.02*i);b[i]=cos(.07*i);}for(int i=0;i<N*4;i++)species[i]=i%2;
  for(int i=0;i<N;i++){spoff[i+1]=spoff[i]+i%5;off[i+1]=np*spoff[i+1];damping[i]=1-.02*i;}
  SetProCapacityHVNACapacityParameters p={np,nr,0};OpenMXGpuTask t={0};t.count=N;t.inputs=6;t.outputs=1;t.kernel=SetProCapacity_HVNA_capacity_kernel;t.parameters=&p;t.parameter_bytes=sizeof(p);
  t.view[0]=(OpenMXGpuTaskView){a,0,0,off,sizeof(float)};t.view[1]=(OpenMXGpuTaskView){b,0,0,off,sizeof(float)};t.view[2]=(OpenMXGpuTaskView){species,0,0,spoff,sizeof(int)};
  t.view[3]=(OpenMXGpuTaskView){energy,sizeof(energy),0};t.view[4]=(OpenMXGpuTaskView){length,sizeof(length),0};t.view[5]=(OpenMXGpuTaskView){damping,0,8};t.view[6]=(OpenMXGpuTaskView){out,0,8};
  for(p.fp32=0;p.fp32<2;p.fp32++){
   fprintf(stderr,"HVNA np=%d nr=%d fp32=%d\n",np,nr,p.fp32);
   for(int r=0;r<N;r++){double total=0;for(size_t k=spoff[r];k<spoff[r+1];k++){double sum=0;int l=0;for(int g=0;g<nr;g++){double val=0;for(int m=0;m<length[g];m++,l++)val+=p.fp32?(double)(a[k*np+l]*b[k*np+l]):(double)a[k*np+l]*b[k*np+l];sum+=energy[nr*species[k]+g]*val;}total+=sum;}expected[r]=damping[r]*total;}
   check(&t,expected,1);
  }
 }
 {enum{G=31};double norm[G],rf[G*2],nlrf[G*2],out[N],derivative[N],expected[2*N];SetNLCapacityRow rows[N];
  for(int i=0;i<G;i++)norm[i]=.04*i;for(int i=0;i<2*G;i++){rf[i]=exp(-.07*i);nlrf[i]=.3+.01*i;}
  for(int r=0;r<N;r++)rows[r]=(SetNLCapacityRow){4,r%3,r%5==0,(r%2)*G,((r+1)%2)*G,.5+.1*r};
  SetNLCapacityParameters p={G,.04};OpenMXGpuTask t={0};t.count=N;t.inputs=4;t.outputs=2;t.kernel=SetNL_capacity_kernel;t.parameters=&p;t.parameter_bytes=sizeof(p);
  t.view[0]=(OpenMXGpuTaskView){rows,0,sizeof(rows[0])};t.view[1]=(OpenMXGpuTaskView){norm,sizeof(norm),0};t.view[2]=(OpenMXGpuTaskView){rf,sizeof(rf),0};t.view[3]=(OpenMXGpuTaskView){nlrf,sizeof(nlrf),0};t.view[4]=(OpenMXGpuTaskView){out,0,8};t.view[5]=(OpenMXGpuTaskView){derivative,0,8};
  /* Analytic j0,j1,j2 and their derivatives provide a recurrence-independent oracle. */
  for(int r=0;r<N;r++){
   long double v=0,d=0;
   for(int i=1;i<G;i++){
    long double x=(long double)norm[i]*rows[r].radius,j0=sinl(x)/x,j1=sinl(x)/(x*x)-cosl(x)/x,j2=3*j1/x-j0;
    long double j=rows[r].l==0?j0:rows[r].l==1?j1:j2;
    long double dj=rows[r].l==0?-j1:rows[r].l==1?j0-2*j1/x:j1-3*j2/x;
    long double weight=(i==G-1?.5L:1)*p.step*norm[i]*norm[i]*rf[rows[r].rf+i]*nlrf[rows[r].nlrf+i];v+=weight*j;d+=weight*norm[i]*dj;
   }
   expected[r]=v;expected[N+r]=rows[r].diag?0:d;
  }
  check(&t,expected,2);
 }

 {enum{G=11,C=3,L=2,R=L*C};double norm[G],rf0[2*G],rf1[2*G],sph[L*G],sphp[L*G],out[4][R],expected[4*R];int off0[C]={0,G,0},off1[C]={G,0,0};
  for(int i=0;i<G;i++)norm[i]=.1+.05*i;
  for(int i=0;i<2*G;i++){rf0[i]=sin(.3*i);rf1[i]=cos(.2*i);sph[i]=.4+.01*i;sphp[i]=-.02*i;}
  OLPKinRadialParameters p={G,G-1,C,1,.05};OpenMXGpuTask t={0};t.count=R;t.inputs=7;t.outputs=4;t.kernel=OLPKin_Radial_Kernel;t.parameters=&p;t.parameter_bytes=sizeof(p);
  t.view[0]=(OpenMXGpuTaskView){norm,sizeof(norm),0};t.view[1]=(OpenMXGpuTaskView){rf0,sizeof(rf0),0};t.view[2]=(OpenMXGpuTaskView){rf1,sizeof(rf1),0};t.view[3]=(OpenMXGpuTaskView){sph,sizeof(sph),0};t.view[4]=(OpenMXGpuTaskView){sphp,sizeof(sphp),0};t.view[5]=(OpenMXGpuTaskView){off0,sizeof(off0),0};t.view[6]=(OpenMXGpuTaskView){off1,sizeof(off1),0};
  for(int c=0;c<4;c++)t.view[7+c]=(OpenMXGpuTaskView){out[c],0,8};
  for(p.h_AN=0;p.h_AN<2;p.h_AN++){
   for(int r=0;r<R;r++)for(int c=0;c<4;c++){
    long double sum=0;
    for(int i=0;i<G;i++){
     long double k=norm[i],radial=(long double)rf0[off0[r%C]+i]*rf1[off1[r%C]+i];
     sum+=(i==0 || i==G-1?.5L:1)*p.h*radial*k*k*(c<2?sph[(r/C)*G+i]:k*sphp[(r/C)*G+i])*(c%2?k*k:1);
    }
    expected[c*R+r]=c>=2 && !p.h_AN?0:(double)sum;
   }
   check(&t,expected,4);
  }
 }
 printf("PASS: %d grid/radial/VNA capacity cases (zero lists, translations, precision, analytic Bessel oracle)\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-grid-recovery-') as folder:
 p=Path(folder);(p/'mpi.h').write_text('typedef int MPI_Comm;\n#define MPI_COMM_NULL (-1)\n')
 (p/'check.c').write_text(prefix+body+functions+main)
 flags=['-Wno-unknown-pragmas','-fsanitize=address,undefined','-fno-sanitize-recover=all']
 subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-g']+flags+['-I'+str(p),'-I'+str(root/'source'),str(p/'check.c'),'-lm','-o',str(p/'check')],check=True)
 subprocess.run([str(p/'check')],check=True,timeout=60)

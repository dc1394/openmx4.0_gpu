/**********************************************************************
  Total_Energy_GPU.c:

     OpenACC helper kernels for grid reductions in Total_Energy.c.

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_core_neutral_grid.h"


#include <string.h>
#include "openmx_gpu_workers.h"
#include "openmx_gpu_task.h"
#include "set_cuda_default_device_from_local_rank.h"

/* Host-backed grid jobs carry their geometry and controls to the helper
   node. No rank-local globals are read by the range kernel. */
typedef struct {
  int kind,spinmax,vna,field,exc0,n2,n3,GNs;
  double vectors[9],origin[3];
} TotalEnergyGridParams;

static int TotalEnergy_GPU_Enabled(void)
{
  return openmx_env_flag_at(OPENMX_ENV_TOTALENERGY_GPU, 1);
}

static void TotalEnergy_Grid_Kernel(size_t first,size_t count,const void *parameters,
                                  void *const view[],int use_gpu)
{
  const TotalEnergyGridParams *p=parameters;
  double *d0=view[0],*d1=view[1],*ad=view[2],*pd0=view[3],*pd1=view[4];
  double *dh=view[5],*v0=view[6],*v1=view[7],*ref=view[8],*vna=view[9],*ef=view[10],*out=view[11];
  double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
  const int spinmax=p->spinmax,pvna=p->vna,field=p->field,exc0=p->exc0;
  if (p->kind==0) {
#pragma acc parallel loop deviceptr(d0,d1,ad,pd0,pd1,dh,v0,v1,ref,vna,ef) reduction(+:s0,s1,s2,s3,s4) if(use_gpu)
    for (size_t i=0;i<count;i++) {
      double den=d0[i]+d1[i],a=ad[i],p0=pd0[i],p1=pd1[i];
      if (!pvna) s0+=den*vna[i];
      if (field) s1+=den*ef[i];
      s2+=(den-2*a)*dh[i];
      s3+=(d0[i]+p0)*v0[i];
      if (spinmax>0) s4+=(d1[i]+p1)*v1[i];
      if (exc0) {
        s3-=(a+p0)*ref[i];
        if (spinmax>0) s4-=(a+p1)*ref[i];
      }
    }
  }
  else if (p->kind==1) {
    const size_t n2=p->n2,n3=p->n3,start=(size_t)p->GNs+first;
    const double x1=p->vectors[0],y1=p->vectors[1],z1=p->vectors[2];
    const double x2=p->vectors[3],y2=p->vectors[4],z2=p->vectors[5];
    const double x3=p->vectors[6],y3=p->vectors[7],z3=p->vectors[8];
    const double ox=p->origin[0],oy=p->origin[1],oz=p->origin[2];
#pragma acc parallel loop deviceptr(d0,d1) reduction(+:s0,s1,s2,s3,s4,s5) if(use_gpu)
    for (size_t i=0;i<count;i++) {
      size_t grid=start+i,a=grid/(n2*n3),b=(grid-a*n2*n3)/n3,c=grid-a*n2*n3-b*n3;
      double x=a*x1+b*x2+c*x3+ox,y=a*y1+b*y2+c*y3+oy,z=a*z1+b*z2+c*z3+oz,den=d0[i]+d1[i];
      s0+=den*x;s1+=den*y;s2+=den*z;s3+=x;s4+=y;s5+=z;
    }
  }
  else {
#pragma acc parallel loop deviceptr(d0,d1,ad,dh,v0,v1) reduction(+:s0,s1,s2,s3) if(use_gpu)
    for (size_t i=0;i<count;i++) {
      s0+=(d0[i]+ad[i])*dh[i];s2+=d0[i]*v0[i];
      if (spinmax>0) {s1+=(d1[i]+ad[i])*dh[i];s3+=d1[i]*v1[i];}
    }
  }
#pragma acc serial deviceptr(out) if(use_gpu)
  {out[0]=s0;out[1]=s1;out[2]=s2;out[3]=s3;out[4]=s4;out[5]=s5;}
}

static void TotalEnergy_Grid_Capacity(int kind,int spinmax,int GNs,double sums[6])
{
  TotalEnergyGridParams params={0};params.kind=kind;params.spinmax=spinmax;
  params.vna=ProExpn_VNA;params.field=E_Field_switch;params.exc0=Exc0_correction_flag;
  params.n2=Ngrid2;params.n3=Ngrid3;params.GNs=GNs;
  for (int i=0;i<3;i++) {
    params.origin[i]=Grid_Origin[i+1];
    for (int j=0;j<3;j++) params.vectors[3*i+j]=gtv[i+1][j+1];
  }
  OpenMXGpuTask t={0};t.count=My_NumGridB_AB;t.parameters=&params;t.parameter_bytes=sizeof(params);
  t.inputs=11;t.outputs=1;t.kernel=TotalEnergy_Grid_Kernel;
  double *data[11]={Density_Grid_B[0],Density_Grid_B[1],ADensity_Grid_B,
    PCCDensity_Grid_B[0],PCCDensity_Grid_B[1],dVHart_Grid_B,Vxc_Grid_B[0],Vxc_Grid_B[1],RefVxc_Grid_B,VNA_Grid_B,VEF_Grid_B};
  for (int i=0;i<11;i++) {
    int used=kind==0 || (kind==1 && i<2) || (kind==2 && (i<3 || (i>=5 && i<=7)));
    if (i==9 && ProExpn_VNA!=0) used=0;
    if (i==10 && E_Field_switch!=1) used=0;
    t.view[i]=(OpenMXGpuTaskView){data[i],0,used?sizeof(double):0};
  }
  t.view[11]=(OpenMXGpuTaskView){sums,6*sizeof(double),0};
  openmx_gpu_task_execute(mpi_comm_level1,&t,TotalEnergy_GPU_Enabled(),
    kind==0?"energy-XC-Hartree":kind==1?"dipole":"CWF-double-counting");
}

void TotalEnergy_EXC_EH1_Grid_OpenACC(int spinmax,double *My_Ena,double *My_Eef,double *My_EH1,double My_EXC[2])
{
  double sums[6];TotalEnergy_Grid_Capacity(0,spinmax,0,sums);
  *My_Ena+=sums[0];*My_Eef+=sums[1];*My_EH1+=sums[2];My_EXC[0]+=sums[3];My_EXC[1]+=sums[4];
}

void TotalEnergy_Dipole_Grid_OpenACC(int GNs,double *My_E_dpx,double *My_E_dpy,double *My_E_dpz,
    double *My_E_dpx_BG,double *My_E_dpy_BG,double *My_E_dpz_BG)
{
  double sums[6];TotalEnergy_Grid_Capacity(1,0,GNs,sums);
  *My_E_dpx+=sums[0];*My_E_dpy+=sums[1];*My_E_dpz+=sums[2];
  *My_E_dpx_BG+=sums[3];*My_E_dpy_BG+=sums[4];*My_E_dpz_BG+=sums[5];
}

void TotalEnergy_CWF_Dc_Grid_OpenACC(int spinmax,double My_dcEH1[2],double My_dcEXC[2])
{
  double sums[6];TotalEnergy_Grid_Capacity(2,spinmax,0,sums);
  for (int spin=0;spin<2;spin++) {My_dcEH1[spin]+=sums[spin];My_dcEXC[spin]+=sums[2+spin];}
}

#pragma acc routine seq
static double TotalEnergy_EH0_VH_AtomF_flat(int spe, int N, double x, double r,
                                            const double *vps_xv_flat,
                                            const double *vps_rv_flat,
                                            const double *vh_atom_flat,
                                            const double *core_charge_flat,
                                            int vps_stride, int vh_stride)
{
  int i;
  double t,dt,xmin,xmax;
  const double *xv,*rv,*yv;

  xv = vps_xv_flat + spe*vps_stride;
  rv = vps_rv_flat + spe*vps_stride;
  yv = vh_atom_flat + spe*vh_stride;

  xmin = xv[0];
  xmax = xv[N-1];

  if (xmax<=x){
    return core_charge_flat[spe]/r;
  }
  else if (r<rv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return a*r*r + b;
  }
  else{
    if (x<xmin) x = xmin;

    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    i = (int)t;
    dt = t - (double)i;

    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt
                 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1];
  }
}


#pragma acc routine seq
static double TotalEnergy_EH0_Dr_VH_AtomF_flat(int spe, int N, double x, double r,
                                               const double *vps_xv_flat,
                                               const double *vps_rv_flat,
                                               const double *vh_atom_flat,
                                               const double *core_charge_flat,
                                               int vps_stride, int vh_stride)
{
  int i;
  double t,dt,xmin,xmax,tmp;
  const double *xv,*rv,*yv;

  xv = vps_xv_flat + spe*vps_stride;
  rv = vps_rv_flat + spe*vps_stride;
  yv = vh_atom_flat + spe*vh_stride;

  xmin = xv[0];
  xmax = xv[N-1];

  if (xmax<=x){
    return -core_charge_flat[spe]/r/r;
  }
  else if (r<rv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,a,b;
    double g1,g2,x1,x2,y1,y2,y12,y22,f,df;

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return 2.0*a*r;
  }
  else{
    if (x<xmin) x = xmin;

    tmp = ((double)N-1.0)/(xmax-xmin);
    t = (x-xmin)*tmp;
    i = (int)t;
    dt = t - (double)i;

    return 0.5*(( 3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  +2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt
                +(yv[i+2]-yv[i]))*tmp/r;
  }
}


typedef struct {int grid_stride,vps_stride,vh_stride;} EH0Params;
static void TotalEnergy_EH0_Kernel(size_t first,size_t count,const void *parameters,void *const view[],int use_gpu)
{
  (void)first;const EH0Params *p=parameters;
  const int grid_stride=p->grid_stride,vps_stride=p->vps_stride,vh_stride=p->vh_stride;
  const int pair_count=(int)count;int pair;
  int *pair_ban=view[0];
  int *pair_wan2=view[1];
  int *pair_has_deriv=view[2];
  double *pair_dis=view[3];
  double *pair_dirx=view[4];
  double *pair_diry=view[5];
  double *pair_dirz=view[6];
  int *tgn_flat=view[7];
  int *vps_n_flat=view[8];
  double *dv_flat=view[9];
  double *core_charge_flat=view[10];
  double *gridx_flat=view[11];
  double *gridy_flat=view[12];
  double *gridz_flat=view[13];
  double *arho_flat=view[14];
  double *wt_flat=view[15];
  double *vps_xv_flat=view[16];
  double *vps_rv_flat=view[17];
  double *vh_atom_flat=view[18];
  double *out0=view[19];
  double *out1=view[20];
  double *out2=view[21];
  double *out3=view[22];
#pragma acc parallel loop gang deviceptr(pair_ban,pair_wan2,pair_has_deriv,pair_dis,pair_dirx,pair_diry,pair_dirz,tgn_flat,vps_n_flat,dv_flat,core_charge_flat,gridx_flat,gridy_flat,gridz_flat,arho_flat,wt_flat,vps_xv_flat,vps_rv_flat,vh_atom_flat,out0,out1,out2,out3) if(use_gpu)
    for (pair=0; pair<pair_count; pair++){
      int n1,ban,wan2,tgn,has_deriv;
      double dis,dirx,diry,dirz,dv,sum,sumr;

      ban = pair_ban[pair];
      wan2 = pair_wan2[pair];
      tgn = tgn_flat[ban];
      has_deriv = pair_has_deriv[pair];
      dis = pair_dis[pair];
      dirx = pair_dirx[pair];
      diry = pair_diry[pair];
      dirz = pair_dirz[pair];
      dv = dv_flat[ban];
      sum = 0.0;
      sumr = 0.0;

#pragma acc loop vector reduction(+:sum,sumr)
      for (n1=0; n1<tgn; n1++){
        int idx;
        double x,y,z,z2,r2,r,xx,rho0,wt,va0,dr_va0;

        idx = ban*grid_stride + n1;
        x = gridx_flat[idx];
        y = gridy_flat[idx];
        z = gridz_flat[idx];
        rho0 = arho_flat[idx];
        wt = wt_flat[idx];
        z2 = z - dis;
        r2 = x*x + y*y + z2*z2;
        r = sqrt(r2);
        xx = 0.5*log(r2);

        if (r<1.0e-10) r = 1.0e-10;

        va0 = TotalEnergy_EH0_VH_AtomF_flat(wan2, vps_n_flat[wan2], xx, r,
                                            vps_xv_flat, vps_rv_flat, vh_atom_flat,
                                            core_charge_flat, vps_stride, vh_stride);

        sum += wt*va0*rho0;

        if (has_deriv && 1.0e-14<r){
          dr_va0 = TotalEnergy_EH0_Dr_VH_AtomF_flat(wan2, vps_n_flat[wan2], xx, r,
                                                    vps_xv_flat, vps_rv_flat, vh_atom_flat,
                                                    core_charge_flat, vps_stride, vh_stride);
          sumr -= wt*rho0*dr_va0*z2/r;
        }
      }

      out0[pair] = sum*dv;
      if (has_deriv){
        sumr = sumr*dv;
        out1[pair] = sumr*dirx;
        out2[pair] = sumr*diry;
        out3[pair] = sumr*dirz;
      }
      else{
        out1[pair] = 0.0;
        out2[pair] = 0.0;
        out3[pair] = 0.0;
      }
    }
}

void TotalEnergy_EH0_TwoCenter_Batch_OpenACC(const OpenMXNeutralGridTables *tables,const OpenMXAtomicViews *atomic, int pair_count, int *pair_ban, int *pair_wan2,
                                             int *pair_has_deriv, double *pair_dis,
                                             double *pair_dirx, double *pair_diry, double *pair_dirz,
                                             double *out0, double *out1, double *out2, double *out3)
{
  int spe,n,i;
  int species_count,grid_stride,vps_stride,vh_stride;
  int grid_count,vps_count,vh_count;
  int *tgn_flat,*vps_n_flat;
  double *gridx_flat,*gridy_flat,*gridz_flat,*arho_flat,*wt_flat,*dv_flat;
  double *vps_xv_flat,*vps_rv_flat,*vh_atom_flat,*core_charge_flat;


  species_count = SpeciesNum;
  grid_stride = Max_TGN_EH0;
  vps_stride = List_YOUSO[22];
  vh_stride = List_YOUSO[22] + 2;
  grid_count = species_count*grid_stride;
  vps_count = species_count*vps_stride;
  vh_count = species_count*vh_stride;

  tgn_flat = (int*)malloc(sizeof(int)*species_count);
  vps_n_flat = (int*)malloc(sizeof(int)*species_count);
  dv_flat = (double*)malloc(sizeof(double)*species_count);
  core_charge_flat = (double*)malloc(sizeof(double)*species_count);
  gridx_flat = (double*)malloc(sizeof(double)*grid_count);
  gridy_flat = (double*)malloc(sizeof(double)*grid_count);
  gridz_flat = (double*)malloc(sizeof(double)*grid_count);
  arho_flat = (double*)malloc(sizeof(double)*grid_count);
  wt_flat = (double*)malloc(sizeof(double)*grid_count);
  vps_xv_flat = (double*)malloc(sizeof(double)*vps_count);
  vps_rv_flat = (double*)malloc(sizeof(double)*vps_count);
  vh_atom_flat = (double*)malloc(sizeof(double)*vh_count);

  if (tgn_flat==NULL || vps_n_flat==NULL || dv_flat==NULL || core_charge_flat==NULL ||
      gridx_flat==NULL || gridy_flat==NULL || gridz_flat==NULL ||
      arho_flat==NULL || wt_flat==NULL ||
      vps_xv_flat==NULL || vps_rv_flat==NULL || vh_atom_flat==NULL){
    printf("TotalEnergy_EH0_TwoCenter_Batch_OpenACC: malloc failed.\n");
    fflush(stdout);
    exit(1);
  }

  for (spe=0; spe<species_count; spe++){
    tgn_flat[spe] = TGN_EH0[spe];
    vps_n_flat[spe] = atomic->species.vps_mesh[spe];
    dv_flat[spe] = dv_EH0[spe];
    core_charge_flat[spe] = atomic->species.core_charge[spe];

    for (n=0; n<grid_stride; n++){
      i = spe*grid_stride + n;
      if (n<TGN_EH0[spe]){
        gridx_flat[i] = openmx_core_neutral_grid_row(tables,0,spe)[n];
        gridy_flat[i] = openmx_core_neutral_grid_row(tables,1,spe)[n];
        gridz_flat[i] = openmx_core_neutral_grid_row(tables,2,spe)[n];
        arho_flat[i] = openmx_core_neutral_grid_row(tables,3,spe)[n];
        wt_flat[i] = openmx_core_neutral_grid_row(tables,4,spe)[n];
      }
      else{
        gridx_flat[i] = 0.0;
        gridy_flat[i] = 0.0;
        gridz_flat[i] = 0.0;
        arho_flat[i] = 0.0;
        wt_flat[i] = 0.0;
      }
    }

    for (n=0; n<vps_stride; n++){
      i = spe*vps_stride + n;
      vps_xv_flat[i] = openmx_radial_row2(&atomic->vps_log_radii,spe)[n];
      vps_rv_flat[i] = openmx_radial_row2(&atomic->vps_radii,spe)[n];
    }

    for (n=0; n<vh_stride; n++){
      i = spe*vh_stride + n;
      vh_atom_flat[i] = openmx_radial_row2(&atomic->hartree_potential,spe)[n];
    }
  }

  EH0Params params={grid_stride,vps_stride,vh_stride};
  OpenMXGpuTask task={0};task.count=pair_count;task.inputs=19;task.outputs=4;
  task.parameters=&params;task.parameter_bytes=sizeof(params);task.kernel=TotalEnergy_EH0_Kernel;
  task.view[0]=(OpenMXGpuTaskView){pair_ban,0,sizeof(int)};
  task.view[1]=(OpenMXGpuTaskView){pair_wan2,0,sizeof(int)};
  task.view[2]=(OpenMXGpuTaskView){pair_has_deriv,0,sizeof(int)};
  task.view[3]=(OpenMXGpuTaskView){pair_dis,0,sizeof(double)};
  task.view[4]=(OpenMXGpuTaskView){pair_dirx,0,sizeof(double)};
  task.view[5]=(OpenMXGpuTaskView){pair_diry,0,sizeof(double)};
  task.view[6]=(OpenMXGpuTaskView){pair_dirz,0,sizeof(double)};
  task.view[7]=(OpenMXGpuTaskView){tgn_flat,(size_t)species_count*sizeof(int),0};
  task.view[8]=(OpenMXGpuTaskView){vps_n_flat,(size_t)species_count*sizeof(int),0};
  task.view[9]=(OpenMXGpuTaskView){dv_flat,(size_t)species_count*sizeof(double),0};
  task.view[10]=(OpenMXGpuTaskView){core_charge_flat,(size_t)species_count*sizeof(double),0};
  task.view[11]=(OpenMXGpuTaskView){gridx_flat,(size_t)grid_count*sizeof(double),0};
  task.view[12]=(OpenMXGpuTaskView){gridy_flat,(size_t)grid_count*sizeof(double),0};
  task.view[13]=(OpenMXGpuTaskView){gridz_flat,(size_t)grid_count*sizeof(double),0};
  task.view[14]=(OpenMXGpuTaskView){arho_flat,(size_t)grid_count*sizeof(double),0};
  task.view[15]=(OpenMXGpuTaskView){wt_flat,(size_t)grid_count*sizeof(double),0};
  task.view[16]=(OpenMXGpuTaskView){vps_xv_flat,(size_t)vps_count*sizeof(double),0};
  task.view[17]=(OpenMXGpuTaskView){vps_rv_flat,(size_t)vps_count*sizeof(double),0};
  task.view[18]=(OpenMXGpuTaskView){vh_atom_flat,(size_t)vh_count*sizeof(double),0};
  task.view[19]=(OpenMXGpuTaskView){out0,0,sizeof(double)};
  task.view[20]=(OpenMXGpuTaskView){out1,0,sizeof(double)};
  task.view[21]=(OpenMXGpuTaskView){out2,0,sizeof(double)};
  task.view[22]=(OpenMXGpuTaskView){out3,0,sizeof(double)};
  openmx_gpu_task_execute(mpi_comm_level1,&task,TotalEnergy_GPU_Enabled(),"energy-EH0-pairs");

  free(tgn_flat);
  free(vps_n_flat);
  free(dv_flat);
  free(core_charge_flat);
  free(gridx_flat);
  free(gridy_flat);
  free(gridz_flat);
  free(arho_flat);
  free(wt_flat);
  free(vps_xv_flat);
  free(vps_rv_flat);
  free(vh_atom_flat);
}


#pragma acc routine seq
static double TotalEnergy_Exc0_XC_CA(double den, int P_switch)
{
  /* device copy of XC_Ceperly_Alder() */

  double dum,rs,coe;
  double Ex,Ec,dEx,dEc;
  double tmp0,tmp1;
  double result;

  if (den<=1.0e-15){
    result = 0.0;
  }
  else{

    coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
    rs = coe*pow(den,-0.3333333333333333333);

    tmp0 = 0.458165293632163/rs;
    Ex = -tmp0;
    dEx = tmp0/rs;

    if (1.0<=rs){
      tmp0 = sqrt(rs);
      dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
      tmp1 = 0.1423/dum;
      Ec = -tmp1;
      dEc = tmp1/dum*(0.52645/tmp0 + 0.3334);
    }
    else{
      tmp0 = log(rs);
      Ec = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
      dEc = 0.0311/rs + 0.0020*tmp0 - 0.0096;
    }

    if      (P_switch==0)
      result = Ex + Ec;
    else if (P_switch==1)
      result = Ex + Ec - 0.33333333333333333333*rs*(dEx + dEc);
    else if (P_switch==2)
      result = 0.3333333333333333333*rs*(dEx + dEc);
    else
      result = -0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx + dEc);
  }

  return result;
}


#pragma acc routine seq
static double TotalEnergy_Exc0_KumoF_flat(int N, double x,
                                          const double *xv, const double *rv,
                                          const double *yv)
{
  /* device copy of KumoF() */

  if (x<xv[0]){

    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df,r;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;

    r = exp(x);

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return a*r*r + b;
  }

  else{

    int i;
    double t,dt;
    double xmin,xmax;

    xmin = xv[0];
    xmax = xv[N-1];
    if (xmax<x) x = xmax;
    if (x<xmin) x = xmin;
    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    i = (int)floor(t);
    dt = t - (double)i;

    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
		  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt
		 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1];
  }
}


#pragma acc routine seq
static double TotalEnergy_Exc0_Dr_KumoF_flat(int N, double x, double r,
                                             const double *xv, const double *rv,
                                             const double *yv)
{
  /* device copy of Dr_KumoF() */

  if (x<xv[0]){

    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,a,b;
    double g1,g2,x1,x2,y1,y2,y12,y22,f,df;

    r = exp(x);

    m = 4;
    rm = rv[m];

    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];

    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;

    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return 2.0*a*r;
  }

  else{

    int i;
    double t,dt,tmp;
    double xmin,xmax;

    xmin = xv[0];
    xmax = xv[N-1];
    if (xmax<x) x = xmax;
    if (x<xmin) x = xmin;

    tmp = ((double)N-1.0)/(xmax-xmin);
    t = (x-xmin)*tmp;
    i = (int)floor(t);
    dt = t - (double)i;

    return 0.5*(( 3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
		  +2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt
		+(yv[i+2]-yv[i]))*tmp/r;
  }
}


typedef struct {int nr,na,nbmax,pao_stride,den_stride,vxc_flag;} Exc0Params;

static void TotalEnergy_Exc0_Kernel(size_t first,size_t count,const void *parameters,void *const view[],int use_gpu)
{
  (void)first;const Exc0Params *p=parameters;
  const int nr=p->nr,na=p->na,nbmax=p->nbmax,pao_stride=p->pao_stride,den_stride=p->den_stride,vxc_flag=p->vxc_flag;
  const int npt_atom=nr*na;
  int *at_fnan=view[0];
  double *at_cx=view[1];
  double *at_cy=view[2];
  double *at_cz=view[3];
  double *at_dr=view[4];
  int *nb_wan=view[5];
  double *nb_x=view[6];
  double *nb_y=view[7];
  double *nb_z=view[8];
  double *nb_rcut2=view[9];
  int *pao_n_flat=view[10];
  double *pao_xv_flat=view[11];
  double *pao_rv_flat=view[12];
  double *den2_flat=view[13];
  double *leb_x=view[14];
  double *leb_y=view[15];
  double *leb_z=view[16];
  double *leb_w=view[17];
  double *abscissa=view[18];
  double *weight=view[19];
  double *energy=view[20];
  double *force_pairs=view[21];
  double *pref=view[22];

#pragma acc parallel loop gang deviceptr(at_fnan,at_cx,at_cy,at_cz,at_dr,nb_wan,nb_x,nb_y,nb_z,nb_rcut2,pao_n_flat,pao_xv_flat,pao_rv_flat,den2_flat,leb_x,leb_y,leb_z,leb_w,abscissa,weight,energy,force_pairs,pref) if(use_gpu)
  for (size_t mc=0;mc<count;mc++) {
    double energy_sum=0;
#pragma acc loop vector reduction(+:energy_sum)
    for (int point=0;point<npt_atom;point++) {
      int ir=point/na,ia=point%na,k,fnan;size_t koff,pt=mc*(size_t)npt_atom+point;
      double r,x0,y0,z0,den,den0,exc0,dexc0,wpt,dr_atom;
      dr_atom = at_dr[mc];
      r = 0.50*(dr_atom*abscissa[ir] + dr_atom);

      x0 = r*leb_x[ia] + at_cx[mc];
      y0 = r*leb_y[ia] + at_cy[mc];
      z0 = r*leb_z[ia] + at_cz[mc];

      koff = (size_t)mc*nbmax;
      fnan = at_fnan[mc];

      den = 0.0;
      den0 = 0.0;

      for (k=0; k<=fnan; k++){
        double dx,dy,dz,r2;

        dx = nb_x[koff+k] - x0;
        dy = nb_y[koff+k] - y0;
        dz = nb_z[koff+k] - z0;
        r2 = dx*dx + dy*dy + dz*dz;

        if (r2<nb_rcut2[koff+k]){
          int wan = nb_wan[koff+k];
          double contrib;

          contrib = TotalEnergy_Exc0_KumoF_flat(pao_n_flat[wan], 0.5*log(r2),
                                                pao_xv_flat + (size_t)wan*pao_stride,
                                                pao_rv_flat + (size_t)wan*pao_stride,
                                                den2_flat + (size_t)wan*den_stride)
                    *(double)vxc_flag;
          den += contrib;
          if (k==0) den0 = contrib;
        }
      }

      exc0 = TotalEnergy_Exc0_XC_CA(den,0);
      dexc0 = TotalEnergy_Exc0_XC_CA(den,3);

      wpt = leb_w[ia]*r*r*weight[ir];

      pref[pt] = wpt*den0*dexc0;
      energy_sum += 2.0*PI*dr_atom*wpt*den0*exc0;    }
    energy[mc]=energy_sum;
  }
#pragma acc parallel loop gang deviceptr(at_fnan,at_cx,at_cy,at_cz,at_dr,nb_wan,nb_x,nb_y,nb_z,nb_rcut2,pao_n_flat,pao_xv_flat,pao_rv_flat,den2_flat,leb_x,leb_y,leb_z,leb_w,abscissa,weight,energy,force_pairs,pref) if(use_gpu)
  for (size_t pp=0;pp<count*(size_t)nbmax;pp++) {
    const size_t mc=pp/(size_t)nbmax;const int k=(int)(pp%(size_t)nbmax);
    force_pairs[3*pp]=force_pairs[3*pp+1]=force_pairs[3*pp+2]=0;
    if (k>0 && k<=at_fnan[mc]) {
          const size_t koff = mc*(size_t)nbmax;
          const int wan = nb_wan[koff+k];
          const int pao_n = pao_n_flat[wan];
          const double rcut2 = nb_rcut2[koff+k];
          const double hx = nb_x[koff+k];
          const double hy = nb_y[koff+k];
          const double hz = nb_z[koff+k];
          const double cx = at_cx[mc];
          const double cy = at_cy[mc];
          const double cz = at_cz[mc];
          const double dr_atom = at_dr[mc];
          const size_t pt0 = (size_t)mc*(size_t)npt_atom;
          double sx = 0.0, sy = 0.0, sz = 0.0;

#pragma acc loop vector reduction(+:sx,sy,sz)
          for (int pt2=0; pt2<npt_atom; pt2++){
            int ir,ia;
            double r,x0,y0,z0,dx,dy,dz,r2;

            ir = pt2/na;
            ia = pt2 - ir*na;

            r = 0.50*(dr_atom*abscissa[ir] + dr_atom);
            x0 = r*leb_x[ia] + cx;
            y0 = r*leb_y[ia] + cy;
            z0 = r*leb_z[ia] + cz;

            dx = hx - x0;
            dy = hy - y0;
            dz = hz - z0;
            r2 = dx*dx + dy*dy + dz*dz;

            if (r2<rcut2){
              double r1,gden0,gscale;

              r1 = sqrt(r2);
              gden0 = TotalEnergy_Exc0_Dr_KumoF_flat(pao_n, 0.5*log(r2), r1,
                                                     pao_xv_flat + (size_t)wan*pao_stride,
                                                     pao_rv_flat + (size_t)wan*pao_stride,
                                                     den2_flat + (size_t)wan*den_stride)
                     *(double)vxc_flag;
              gscale = pref[pt0 + (size_t)pt2]*gden0/r1;

              sx += gscale*dx;
              sy += gscale*dy;
              sz += gscale*dz;
            }
          }

      force_pairs[3*pp]=2.0*PI*dr_atom*sx;
      force_pairs[3*pp+1]=2.0*PI*dr_atom*sy;
      force_pairs[3*pp+2]=2.0*PI*dr_atom*sz;
    }
  }
}

void TotalEnergy_Exc0_Batch_OpenACC(const OpenMXAtomicViews *atomic, int na,double **leb,double *sum)
{
  int nbmax=1;
  for (int mc=1;mc<=Matomnum;mc++) if (nbmax<FNAN[M2G[mc]]+1) nbmax=FNAN[M2G[mc]]+1;
  Exc0Params p={CoarseGL_Mesh,na,nbmax,List_YOUSO[21],List_YOUSO[21]+2,F_Vxc_flag};
  OpenMXGpuTask t={0};t.count=Matomnum;t.inputs=20;t.outputs=2;t.scratch=1;
  t.parameters=&p;t.parameter_bytes=sizeof(p);t.kernel=TotalEnergy_Exc0_Kernel;
  t.view[0]=(OpenMXGpuTaskView){NULL,0,sizeof(int)};
  t.view[1]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  t.view[2]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  t.view[3]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  t.view[4]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  t.view[5]=(OpenMXGpuTaskView){NULL,0,(size_t)nbmax*sizeof(int)};
  t.view[6]=(OpenMXGpuTaskView){NULL,0,(size_t)nbmax*sizeof(double)};
  t.view[7]=(OpenMXGpuTaskView){NULL,0,(size_t)nbmax*sizeof(double)};
  t.view[8]=(OpenMXGpuTaskView){NULL,0,(size_t)nbmax*sizeof(double)};
  t.view[9]=(OpenMXGpuTaskView){NULL,0,(size_t)nbmax*sizeof(double)};
  t.view[10]=(OpenMXGpuTaskView){NULL,(size_t)SpeciesNum*sizeof(int),0};
  t.view[11]=(OpenMXGpuTaskView){NULL,(size_t)SpeciesNum*p.pao_stride*sizeof(double),0};
  t.view[12]=(OpenMXGpuTaskView){NULL,(size_t)SpeciesNum*p.pao_stride*sizeof(double),0};
  t.view[13]=(OpenMXGpuTaskView){NULL,(size_t)SpeciesNum*p.den_stride*sizeof(double),0};
  t.view[14]=(OpenMXGpuTaskView){NULL,(size_t)na*sizeof(double),0};
  t.view[15]=(OpenMXGpuTaskView){NULL,(size_t)na*sizeof(double),0};
  t.view[16]=(OpenMXGpuTaskView){NULL,(size_t)na*sizeof(double),0};
  t.view[17]=(OpenMXGpuTaskView){NULL,(size_t)na*sizeof(double),0};
  t.view[18]=(OpenMXGpuTaskView){NULL,(size_t)p.nr*sizeof(double),0};
  t.view[19]=(OpenMXGpuTaskView){NULL,(size_t)p.nr*sizeof(double),0};
  t.view[20]=(OpenMXGpuTaskView){NULL,0,sizeof(double)};
  t.view[21]=(OpenMXGpuTaskView){NULL,0,(size_t)3*nbmax*sizeof(double)};
  t.view[22]=(OpenMXGpuTaskView){NULL,0,(size_t)p.nr*na*sizeof(double)};
  for (int i=0;i<22;i++) {
    size_t bytes=t.view[i].item_bytes?t.count*t.view[i].item_bytes:t.view[i].bytes;
    t.view[i].data=calloc(bytes?bytes:1,1);
    if (!t.view[i].data) {fprintf(stderr,"Exc0: host table allocation failed.\n");MPI_Abort(mpi_comm_level1,1);}
  }
  int *at_fnan=t.view[0].data;
  double *at_cx=t.view[1].data;
  double *at_cy=t.view[2].data;
  double *at_cz=t.view[3].data;
  double *at_dr=t.view[4].data;
  int *nb_wan=t.view[5].data;
  double *nb_x=t.view[6].data;
  double *nb_y=t.view[7].data;
  double *nb_z=t.view[8].data;
  double *nb_rcut2=t.view[9].data;
  int *pao_n_flat=t.view[10].data;
  double *pao_xv_flat=t.view[11].data;
  double *pao_rv_flat=t.view[12].data;
  double *den2_flat=t.view[13].data;
  double *leb_x=t.view[14].data;
  double *leb_y=t.view[15].data;
  double *leb_z=t.view[16].data;
  double *leb_w=t.view[17].data;
  double *abscissa=t.view[18].data;
  double *weight=t.view[19].data;
  double *energy=t.view[20].data;
  double *force_pairs=t.view[21].data;
  for (int spe=0;spe<SpeciesNum;spe++) {
    pao_n_flat[spe]=atomic->species.pao_mesh[spe];
    memcpy(pao_xv_flat+(size_t)spe*p.pao_stride,openmx_radial_row2(&atomic->pao_log_radii,spe),(size_t)p.pao_stride*sizeof(double));
    memcpy(pao_rv_flat+(size_t)spe*p.pao_stride,openmx_radial_row2(&atomic->pao_radii,spe),(size_t)p.pao_stride*sizeof(double));
    memcpy(den2_flat+(size_t)spe*p.den_stride,openmx_radial_row2(&atomic->density_backup,spe),(size_t)p.den_stride*sizeof(double));
  }
  for (int mc=0;mc<Matomnum;mc++) {
    int gc=M2G[mc+1];at_fnan[mc]=FNAN[gc];
    at_cx[mc]=Gxyz[gc][1];at_cy[mc]=Gxyz[gc][2];at_cz[mc]=Gxyz[gc][3];at_dr[mc]=atomic->species.cutoff[WhatSpecies[gc]];
    for (int h=0;h<=FNAN[gc];h++) {
      int gh=natn[gc][h],rn=ncn[gc][h],wan=WhatSpecies[gh];size_t k=(size_t)mc*nbmax+h;
      nb_wan[k]=wan;nb_x[k]=Gxyz[gh][1]+atv[rn][1];nb_y[k]=Gxyz[gh][2]+atv[rn][2];nb_z[k]=Gxyz[gh][3]+atv[rn][3];
      nb_rcut2[k]=atomic->species.cutoff[wan]*atomic->species.cutoff[wan];
    }
  }
  for (int i=0;i<na;i++) {leb_x[i]=leb[i][0];leb_y[i]=leb[i][1];leb_z[i]=leb[i][2];leb_w[i]=leb[i][3];}
  memcpy(abscissa,CoarseGL_Abscissae,(size_t)p.nr*sizeof(double));
  memcpy(weight,CoarseGL_Weight,(size_t)p.nr*sizeof(double));
  openmx_gpu_task_execute(mpi_comm_level1,&t,TotalEnergy_GPU_Enabled(),"energy-Exc0-two-pass");
  *sum=0;
  for (int mc=0;mc<Matomnum;mc++) {
    int gc=M2G[mc+1];*sum+=energy[mc];
    for (int h=1;h<=FNAN[gc];h++) {
      int gh=natn[gc][h];size_t k=(size_t)mc*nbmax+h;
      for (int axis=0;axis<3;axis++) {Gxyz[gh][41+axis]+=force_pairs[3*k+axis];Gxyz[gc][41+axis]-=force_pairs[3*k+axis];}
    }
  }
  for (int i=0;i<22;i++) free(t.view[i].data);
}

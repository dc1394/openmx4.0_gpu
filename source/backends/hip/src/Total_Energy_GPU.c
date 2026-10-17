/**********************************************************************
  Total_Energy_GPU.c:

     OpenMP helper kernels for grid reductions in Total_Energy.c.

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"


#include <string.h>
#include "openmx_gpu_workers.h"
#include "openmx_gpu_task.h"
#include "set_hip_default_device_from_local_rank.h"

void AMD_TotalEnergy_EXC_EH1_Grid_OpenMP(int spinmax, double *My_Ena, double *My_Eef,
                                      double *My_EH1, double My_EXC[2])
{
  int BN,grid_count,vna_grid_count,vef_grid_count;
  double local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1;
  double *Density_Grid_B0,*Density_Grid_B1;
  double *PCCDensity_Grid_B0,*PCCDensity_Grid_B1;
  double *Vxc_Grid_B0,*Vxc_Grid_B1;

  grid_count = My_NumGridB_AB;
  vna_grid_count = (ProExpn_VNA==0) ? grid_count : 0;
  vef_grid_count = (E_Field_switch==1) ? grid_count : 0;
  Density_Grid_B0 = Density_Grid_B[0];
  Density_Grid_B1 = Density_Grid_B[1];
  PCCDensity_Grid_B0 = PCCDensity_Grid_B[0];
  PCCDensity_Grid_B1 = PCCDensity_Grid_B[1];
  Vxc_Grid_B0 = Vxc_Grid_B[0];
  Vxc_Grid_B1 = Vxc_Grid_B[1];

  local_Ena = *My_Ena;
  local_Eef = *My_Eef;
  local_EH1 = *My_EH1;
  local_EXC0 = My_EXC[0];
  local_EXC1 = My_EXC[1];

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count], \
                        ADensity_Grid_B[0:grid_count], PCCDensity_Grid_B0[0:grid_count], \
                        PCCDensity_Grid_B1[0:grid_count], dVHart_Grid_B[0:grid_count], \
                        Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count], \
                        RefVxc_Grid_B[0:grid_count], VNA_Grid_B[0:vna_grid_count], \
                        VEF_Grid_B[0:vef_grid_count])
  {
#pragma omp target teams distribute parallel for num_teams(128) thread_limit(256) reduction(+:local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1) map(tofrom:local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1)
    for (BN=0; BN<grid_count; BN++){
      double sden0,sden1,tden,aden,pden0,pden1,refvxc;

      sden0 = Density_Grid_B0[BN];
      sden1 = Density_Grid_B1[BN];
      tden = sden0 + sden1;
      aden = ADensity_Grid_B[BN];
      pden0 = PCCDensity_Grid_B0[BN];
      pden1 = PCCDensity_Grid_B1[BN];
      refvxc = RefVxc_Grid_B[BN];

      if (ProExpn_VNA==0){
        local_Ena += tden*VNA_Grid_B[BN];
      }
      if (E_Field_switch==1){
        local_Eef += tden*VEF_Grid_B[BN];
      }

      local_EH1 += (tden - 2.0*aden)*dVHart_Grid_B[BN];

      if (Exc0_correction_flag==1){
        local_EXC0 += (sden0+pden0)*Vxc_Grid_B0[BN] - (aden+pden0)*refvxc;
        if (0<spinmax){
          local_EXC1 += (sden1+pden1)*Vxc_Grid_B1[BN] - (aden+pden1)*refvxc;
        }
      }
      else{
        local_EXC0 += (sden0+pden0)*Vxc_Grid_B0[BN];
        if (0<spinmax){
          local_EXC1 += (sden1+pden1)*Vxc_Grid_B1[BN];
        }
      }
    }
  }

  *My_Ena = local_Ena;
  *My_Eef = local_Eef;
  *My_EH1 = local_EH1;
  My_EXC[0] = local_EXC0;
  My_EXC[1] = local_EXC1;
}


void AMD_TotalEnergy_Dipole_Grid_OpenMP(int GNs, double *My_E_dpx, double *My_E_dpy,
                                     double *My_E_dpz, double *My_E_dpx_BG,
                                     double *My_E_dpy_BG, double *My_E_dpz_BG)
{
  int BN,grid_count;
  double local_E_dpx,local_E_dpy,local_E_dpz;
  double local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG;
  double gtv11,gtv12,gtv13,gtv21,gtv22,gtv23,gtv31,gtv32,gtv33;
  double grid_origin1,grid_origin2,grid_origin3;
  double *Density_Grid_B0,*Density_Grid_B1;

  grid_count = My_NumGridB_AB;
  Density_Grid_B0 = Density_Grid_B[0];
  Density_Grid_B1 = Density_Grid_B[1];
  gtv11 = gtv[1][1];
  gtv12 = gtv[1][2];
  gtv13 = gtv[1][3];
  gtv21 = gtv[2][1];
  gtv22 = gtv[2][2];
  gtv23 = gtv[2][3];
  gtv31 = gtv[3][1];
  gtv32 = gtv[3][2];
  gtv33 = gtv[3][3];
  grid_origin1 = Grid_Origin[1];
  grid_origin2 = Grid_Origin[2];
  grid_origin3 = Grid_Origin[3];

  local_E_dpx = *My_E_dpx;
  local_E_dpy = *My_E_dpy;
  local_E_dpz = *My_E_dpz;
  local_E_dpx_BG = *My_E_dpx_BG;
  local_E_dpy_BG = *My_E_dpy_BG;
  local_E_dpz_BG = *My_E_dpz_BG;

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count])
  {
#pragma omp target teams distribute parallel for num_teams(128) thread_limit(256) reduction(+:local_E_dpx,local_E_dpy,local_E_dpz,local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG) map(tofrom:local_E_dpx,local_E_dpy,local_E_dpz,local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG)
    for (BN=0; BN<grid_count; BN++){
      int GN,n1,n2,n3;
      double x,y,z,den;

      GN = BN + GNs;
      n1 = GN/(Ngrid2*Ngrid3);
      n2 = (GN - n1*Ngrid2*Ngrid3)/Ngrid3;
      n3 = GN - n1*Ngrid2*Ngrid3 - n2*Ngrid3;

      x = (double)n1*gtv11 + (double)n2*gtv21 + (double)n3*gtv31 + grid_origin1;
      y = (double)n1*gtv12 + (double)n2*gtv22 + (double)n3*gtv32 + grid_origin2;
      z = (double)n1*gtv13 + (double)n2*gtv23 + (double)n3*gtv33 + grid_origin3;
      den = Density_Grid_B0[BN] + Density_Grid_B1[BN];

      local_E_dpx += den*x;
      local_E_dpy += den*y;
      local_E_dpz += den*z;
      local_E_dpx_BG += x;
      local_E_dpy_BG += y;
      local_E_dpz_BG += z;
    }
  }

  *My_E_dpx = local_E_dpx;
  *My_E_dpy = local_E_dpy;
  *My_E_dpz = local_E_dpz;
  *My_E_dpx_BG = local_E_dpx_BG;
  *My_E_dpy_BG = local_E_dpy_BG;
  *My_E_dpz_BG = local_E_dpz_BG;
}


void AMD_TotalEnergy_CWF_Dc_Grid_OpenMP(int spinmax, double My_dcEH1[2], double My_dcEXC[2])
{
  int BN,grid_count;
  double local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1;
  double *Density_Grid_B0,*Density_Grid_B1;
  double *Vxc_Grid_B0,*Vxc_Grid_B1;

  grid_count = My_NumGridB_AB;
  Density_Grid_B0 = Density_Grid_B[0];
  Density_Grid_B1 = Density_Grid_B[1];
  Vxc_Grid_B0 = Vxc_Grid_B[0];
  Vxc_Grid_B1 = Vxc_Grid_B[1];

  local_dcEH10 = My_dcEH1[0];
  local_dcEH11 = My_dcEH1[1];
  local_dcEXC0 = My_dcEXC[0];
  local_dcEXC1 = My_dcEXC[1];

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count], \
                        ADensity_Grid_B[0:grid_count], dVHart_Grid_B[0:grid_count], \
                        Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count])
  {
#pragma omp target teams distribute parallel for num_teams(128) thread_limit(256) reduction(+:local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1) map(tofrom:local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1)
    for (BN=0; BN<grid_count; BN++){
      double sden0,sden1,aden,dvhart;

      sden0 = Density_Grid_B0[BN];
      sden1 = Density_Grid_B1[BN];
      aden = ADensity_Grid_B[BN];
      dvhart = dVHart_Grid_B[BN];

      local_dcEH10 += (sden0 + aden)*dvhart;
      local_dcEXC0 += sden0*Vxc_Grid_B0[BN];

      if (0<spinmax){
        local_dcEH11 += (sden1 + aden)*dvhart;
        local_dcEXC1 += sden1*Vxc_Grid_B1[BN];
      }
    }
  }

  My_dcEH1[0] = local_dcEH10;
  My_dcEH1[1] = local_dcEH11;
  My_dcEXC[0] = local_dcEXC0;
  My_dcEXC[1] = local_dcEXC1;
}



#pragma omp declare target
static double AMD_TotalEnergy_EH0_VH_AtomF_flat(int spe, int N, double x, double r,
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
#pragma omp end declare target



#pragma omp declare target
static double AMD_TotalEnergy_EH0_Dr_VH_AtomF_flat(int spe, int N, double x, double r,
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
#pragma omp end declare target


static void AMD_TotalEnergy_EH0_TwoCenter_Batch_Run(int pair_count, int *pair_ban, int *pair_wan2,
                                             int *pair_has_deriv, double *pair_dis,
                                             double *pair_dirx, double *pair_diry, double *pair_dirz,
                                             double *out0, double *out1, double *out2, double *out3)
{
  int spe,n,i,pair;
  int species_count,grid_stride,vps_stride,vh_stride;
  int grid_count,vps_count,vh_count;
  int *tgn_flat,*vps_n_flat;
  double *gridx_flat,*gridy_flat,*gridz_flat,*arho_flat,*wt_flat,*dv_flat;
  double *vps_xv_flat,*vps_rv_flat,*vh_atom_flat,*core_charge_flat;

  if (pair_count<=0) return;

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
    printf("AMD_TotalEnergy_EH0_TwoCenter_Batch_OpenMP: malloc failed.\n");
    fflush(stdout);
    exit(1);
  }

  for (spe=0; spe<species_count; spe++){
    tgn_flat[spe] = TGN_EH0[spe];
    vps_n_flat[spe] = Spe_Num_Mesh_VPS[spe];
    dv_flat[spe] = dv_EH0[spe];
    core_charge_flat[spe] = Spe_Core_Charge[spe];

    for (n=0; n<grid_stride; n++){
      i = spe*grid_stride + n;
      if (n<TGN_EH0[spe]){
        gridx_flat[i] = GridX_EH0[spe][n];
        gridy_flat[i] = GridY_EH0[spe][n];
        gridz_flat[i] = GridZ_EH0[spe][n];
        arho_flat[i] = Arho_EH0[spe][n];
        wt_flat[i] = Wt_EH0[spe][n];
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
      vps_xv_flat[i] = Spe_VPS_XV[spe][n];
      vps_rv_flat[i] = Spe_VPS_RV[spe][n];
    }

    for (n=0; n<vh_stride; n++){
      i = spe*vh_stride + n;
      vh_atom_flat[i] = Spe_VH_Atom[spe][n];
    }
  }

#pragma omp target data map(to:pair_ban[0:pair_count], pair_wan2[0:pair_count], \
                        pair_has_deriv[0:pair_count], pair_dis[0:pair_count], \
                        pair_dirx[0:pair_count], pair_diry[0:pair_count], pair_dirz[0:pair_count], \
                        tgn_flat[0:species_count], vps_n_flat[0:species_count], \
                        dv_flat[0:species_count], core_charge_flat[0:species_count], \
                        gridx_flat[0:grid_count], gridy_flat[0:grid_count], \
                        gridz_flat[0:grid_count], arho_flat[0:grid_count], wt_flat[0:grid_count], \
                        vps_xv_flat[0:vps_count], vps_rv_flat[0:vps_count], vh_atom_flat[0:vh_count]) map(from:out0[0:pair_count], out1[0:pair_count], out2[0:pair_count], out3[0:pair_count])
  {
#pragma omp target teams distribute thread_limit(128)
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

      /* One team owns a pair; its lanes share the long radial integration.
         The outer construct has no parallel region, so this is a single
         level of parallelism, including for the derivative reduction. */
#pragma omp parallel for reduction(+:sum,sumr)
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

        va0 = AMD_TotalEnergy_EH0_VH_AtomF_flat(wan2, vps_n_flat[wan2], xx, r,
                                            vps_xv_flat, vps_rv_flat, vh_atom_flat,
                                            core_charge_flat, vps_stride, vh_stride);

        sum += wt*va0*rho0;

        if (has_deriv && 1.0e-14<r){
          dr_va0 = AMD_TotalEnergy_EH0_Dr_VH_AtomF_flat(wan2, vps_n_flat[wan2], xx, r,
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



/* A large pair batch already fills a GPU. Limit concurrent long kernels
   and target-data copies from separate MPI processes sharing that device.
   Small batches and separate physical GPUs retain concurrent dispatch. */
void AMD_TotalEnergy_EH0_TwoCenter_Batch_OpenMP(int pair_count, int *pair_ban, int *pair_wan2,
                                             int *pair_has_deriv, double *pair_dis,
                                             double *pair_dirx, double *pair_diry, double *pair_dirz,
                                             double *out0, double *out1, double *out2, double *out3)
{
  MPI_Comm node,group;
  int node_rank,node_size,rank,size,device,color,max_pairs,concurrency;
  char bus[32] = {0}, *buses;
  const char *env = getenv("OPENMX_EH0_GPU_CONCURRENCY");

  if (env != NULL && atoi(env)==0) {
    AMD_TotalEnergy_EH0_TwoCenter_Batch_Run(pair_count,pair_ban,pair_wan2,pair_has_deriv,
                                     pair_dis,pair_dirx,pair_diry,pair_dirz,out0,out1,out2,out3);
    return;
  }
  MPI_Comm_split_type(mpi_comm_level1,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node);
  MPI_Comm_rank(node,&node_rank);
  MPI_Comm_size(node,&node_size);
  buses = (char*)calloc((size_t)node_size,sizeof(bus));
  if (buses==NULL) MPI_Abort(mpi_comm_level1,1);
  if (hipGetDevice(&device)==hipSuccess)
    (void)hipDeviceGetPCIBusId(bus,sizeof(bus),device);
  MPI_Allgather(bus,sizeof(bus),MPI_CHAR,buses,sizeof(bus),MPI_CHAR,node);
  color = 0;
  while (color<node_rank && strcmp(bus,buses+(size_t)color*sizeof(bus))!=0) ++color;
  MPI_Comm_split(node,color,node_rank,&group);
  free(buses);
  MPI_Comm_free(&node);
  MPI_Comm_rank(group,&rank);
  MPI_Comm_size(group,&size);
  MPI_Allreduce(&pair_count,&max_pairs,1,MPI_INT,MPI_MAX,group);
  concurrency = max_pairs>=8192 ? 1 : size;
  if (env != NULL && atoi(env)>0) concurrency = atoi(env);
  if (concurrency>size) concurrency = size;
  if (rank==0 && concurrency<size && getenv("OPENMX_GPU_VERBOSE")!=NULL) {
    printf("EH0 GPU %s: %d ranks, %d active per turn, maximum %d pairs per rank.\n",
           bus,size,concurrency,max_pairs);
    fflush(stdout);
  }
  for (int first=0; first<size; first+=concurrency) {
    if (first<=rank && rank<first+concurrency)
      AMD_TotalEnergy_EH0_TwoCenter_Batch_Run(pair_count,pair_ban,pair_wan2,pair_has_deriv,
                                       pair_dis,pair_dirx,pair_diry,pair_dirz,out0,out1,out2,out3);
    if (concurrency<size) MPI_Barrier(group);
  }
  MPI_Comm_free(&group);
}

#pragma omp declare target
static double AMD_TotalEnergy_Exc0_XC_CA(double den, int P_switch)
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
#pragma omp end declare target



#pragma omp declare target
static double AMD_TotalEnergy_Exc0_KumoF_flat(int N, double x,
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
#pragma omp end declare target



#pragma omp declare target
static double AMD_TotalEnergy_Exc0_Dr_KumoF_flat(int N, double x, double r,
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
#pragma omp end declare target


void AMD_TotalEnergy_Exc0_Batch_OpenMP(int Num_Leb, double **Leb_Grid_XYZW, double *sum_out)
{
  /* fine-mesh Exc^0 correction of Calc_EXC_EH1: for every local atom,
     integrate den0*exc(den) over the atom-centred Gauss-Legendre x
     Lebedev grid, together with the density-gradient force terms.
     One flattened point kernel accumulates the energy and stores the
     per-point force prefactor; a second pair kernel contracts the
     prefactors with the neighbour density gradients. */

  int spe,Mc_AN,Gc_AN,h_AN,i,n;
  int nr,na,npt_atom,nb_total;
  size_t npt_total;
  int pao_stride,den_stride;
  int vxc_flag;
  int *pao_n_flat,*at_fnan,*nb_off,*nb_wan;
  double *pao_xv_flat,*pao_rv_flat,*den2_flat;
  double *at_cx,*at_cy,*at_cz,*at_dr;
  double *nb_x,*nb_y,*nb_z,*nb_rcut2;
  double *leb_x,*leb_y,*leb_z,*leb_w;
  double *pref;
  double energy_sum;

  *sum_out = 0.0;
  if (Matomnum<1) return;

  nr = CoarseGL_Mesh;
  na = Num_Leb;
  npt_atom = nr*na;
  npt_total = (size_t)Matomnum*(size_t)npt_atom;

  pao_stride = List_YOUSO[21];
  den_stride = List_YOUSO[21] + 2;
  vxc_flag = F_Vxc_flag;

  /* species tables */

  pao_n_flat = (int*)malloc(sizeof(int)*SpeciesNum);
  pao_xv_flat = (double*)malloc(sizeof(double)*(size_t)SpeciesNum*pao_stride);
  pao_rv_flat = (double*)malloc(sizeof(double)*(size_t)SpeciesNum*pao_stride);
  den2_flat = (double*)malloc(sizeof(double)*(size_t)SpeciesNum*den_stride);

  /* local atoms and their neighbour lists */

  at_fnan = (int*)malloc(sizeof(int)*(Matomnum+1));
  nb_off = (int*)malloc(sizeof(int)*(Matomnum+1));
  at_cx = (double*)malloc(sizeof(double)*(Matomnum+1));
  at_cy = (double*)malloc(sizeof(double)*(Matomnum+1));
  at_cz = (double*)malloc(sizeof(double)*(Matomnum+1));
  at_dr = (double*)malloc(sizeof(double)*(Matomnum+1));

  nb_total = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    nb_total += FNAN[M2G[Mc_AN]] + 1;
  }

  nb_wan = (int*)malloc(sizeof(int)*nb_total);
  nb_x = (double*)malloc(sizeof(double)*nb_total);
  nb_y = (double*)malloc(sizeof(double)*nb_total);
  nb_z = (double*)malloc(sizeof(double)*nb_total);
  nb_rcut2 = (double*)malloc(sizeof(double)*nb_total);

  leb_x = (double*)malloc(sizeof(double)*na);
  leb_y = (double*)malloc(sizeof(double)*na);
  leb_z = (double*)malloc(sizeof(double)*na);
  leb_w = (double*)malloc(sizeof(double)*na);

  /* device-resident via create(); the host allocation is only address space */
  pref = (double*)malloc(sizeof(double)*npt_total);

  if (pao_n_flat==NULL || pao_xv_flat==NULL || pao_rv_flat==NULL || den2_flat==NULL ||
      at_fnan==NULL || nb_off==NULL || at_cx==NULL || at_cy==NULL || at_cz==NULL ||
      at_dr==NULL || nb_wan==NULL || nb_x==NULL || nb_y==NULL || nb_z==NULL ||
      nb_rcut2==NULL || leb_x==NULL || leb_y==NULL || leb_z==NULL || leb_w==NULL ||
      pref==NULL){
    printf("AMD_TotalEnergy_Exc0_Batch_OpenMP: malloc failed.\n");
    fflush(stdout);
    exit(1);
  }

  for (spe=0; spe<SpeciesNum; spe++){
    pao_n_flat[spe] = Spe_Num_Mesh_PAO[spe];
    for (n=0; n<pao_stride; n++){
      pao_xv_flat[(size_t)spe*pao_stride+n] = Spe_PAO_XV[spe][n];
      pao_rv_flat[(size_t)spe*pao_stride+n] = Spe_PAO_RV[spe][n];
    }
    for (n=0; n<den_stride; n++){
      den2_flat[(size_t)spe*den_stride+n] = Spe_Atomic_Den2[spe][n];
    }
  }

  n = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Cwan;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    at_fnan[Mc_AN] = FNAN[Gc_AN];
    nb_off[Mc_AN] = n;
    at_cx[Mc_AN] = Gxyz[Gc_AN][1];
    at_cy[Mc_AN] = Gxyz[Gc_AN][2];
    at_cz[Mc_AN] = Gxyz[Gc_AN][3];
    at_dr[Mc_AN] = Spe_Atom_Cut1[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      int Gh_AN = natn[Gc_AN][h_AN];
      int Rn = ncn[Gc_AN][h_AN];
      int Hwan = WhatSpecies[Gh_AN];

      nb_wan[n] = Hwan;
      nb_x[n] = Gxyz[Gh_AN][1] + atv[Rn][1];
      nb_y[n] = Gxyz[Gh_AN][2] + atv[Rn][2];
      nb_z[n] = Gxyz[Gh_AN][3] + atv[Rn][3];
      nb_rcut2[n] = Spe_Atom_Cut1[Hwan]*Spe_Atom_Cut1[Hwan];
      n++;
    }
  }

  for (i=0; i<na; i++){
    leb_x[i] = Leb_Grid_XYZW[i][0];
    leb_y[i] = Leb_Grid_XYZW[i][1];
    leb_z[i] = Leb_Grid_XYZW[i][2];
    leb_w[i] = Leb_Grid_XYZW[i][3];
  }

  energy_sum = 0.0;

#pragma omp target data map(to:pao_n_flat[0:SpeciesNum], \
                        pao_xv_flat[0:(size_t)SpeciesNum*pao_stride], \
                        pao_rv_flat[0:(size_t)SpeciesNum*pao_stride], \
                        den2_flat[0:(size_t)SpeciesNum*den_stride], \
                        at_fnan[0:Matomnum+1], nb_off[0:Matomnum+1], \
                        at_cx[0:Matomnum+1], at_cy[0:Matomnum+1], \
                        at_cz[0:Matomnum+1], at_dr[0:Matomnum+1], \
                        nb_wan[0:nb_total], nb_x[0:nb_total], nb_y[0:nb_total], \
                        nb_z[0:nb_total], nb_rcut2[0:nb_total], \
                        leb_x[0:na], leb_y[0:na], leb_z[0:na], leb_w[0:na], \
                        CoarseGL_Abscissae[0:nr], CoarseGL_Weight[0:nr]) map(from:pref[0:npt_total])
  {

    /* pass 1: energy and per-point force prefactors */

#pragma omp target teams distribute parallel for thread_limit(128) reduction(+:energy_sum) map(tofrom:energy_sum)
    for (size_t pt=0; pt<npt_total; pt++){
      int mc,ir,ia,rem,k,koff,fnan;
      double r,x0,y0,z0,den,den0,exc0,dexc0,wpt,dr_atom;

      mc = (int)(pt/(size_t)npt_atom) + 1;
      rem = (int)(pt - (size_t)(mc-1)*(size_t)npt_atom);
      ir = rem/na;
      ia = rem - ir*na;

      dr_atom = at_dr[mc];
      r = 0.50*(dr_atom*CoarseGL_Abscissae[ir] + dr_atom);

      x0 = r*leb_x[ia] + at_cx[mc];
      y0 = r*leb_y[ia] + at_cy[mc];
      z0 = r*leb_z[ia] + at_cz[mc];

      koff = nb_off[mc];
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

          contrib = AMD_TotalEnergy_Exc0_KumoF_flat(pao_n_flat[wan], 0.5*log(r2),
                                                pao_xv_flat + (size_t)wan*pao_stride,
                                                pao_rv_flat + (size_t)wan*pao_stride,
                                                den2_flat + (size_t)wan*den_stride)
                    *(double)vxc_flag;
          den += contrib;
          if (k==0) den0 = contrib;
        }
      }

      exc0 = AMD_TotalEnergy_Exc0_XC_CA(den,0);
      dexc0 = AMD_TotalEnergy_Exc0_XC_CA(den,3);

      wpt = leb_w[ia]*r*r*CoarseGL_Weight[ir];

      pref[pt] = wpt*den0*dexc0;
      energy_sum += 2.0*PI*dr_atom*wpt*den0*exc0;
    }
  }

  /* pass 2: contract the prefactors with the neighbour gradients.
     Keep this in a separate target data region.  ROCm clang 18 can emit
     invalid LLVM IR when this region is nested in the pass-1 target-data
     region and uses present mappings for the shared arrays. */

    {
      int nitems = nb_total - Matomnum; /* h_AN != 0 entries */
      int nitems_alloc = nitems==0 ? 1 : nitems;
      int *item_mc,*item_k;
      double *item_fx,*item_fy,*item_fz;
      int p;

      item_mc = (int*)malloc(sizeof(int)*nitems_alloc);
      item_k = (int*)malloc(sizeof(int)*nitems_alloc);
      item_fx = (double*)malloc(sizeof(double)*nitems_alloc);
      item_fy = (double*)malloc(sizeof(double)*nitems_alloc);
      item_fz = (double*)malloc(sizeof(double)*nitems_alloc);

      if (item_mc==NULL || item_k==NULL || item_fx==NULL || item_fy==NULL || item_fz==NULL){
        printf("AMD_TotalEnergy_Exc0_Batch_OpenMP: malloc failed (pass 2).\n");
        fflush(stdout);
        exit(1);
      }

      p = 0;
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        for (h_AN=1; h_AN<=at_fnan[Mc_AN]; h_AN++){
          item_mc[p] = Mc_AN;
          item_k[p] = h_AN;
          p++;
        }
      }

#pragma omp target data map(to:item_mc[0:nitems_alloc], item_k[0:nitems_alloc], \
            pao_n_flat[0:SpeciesNum], pao_xv_flat[0:(size_t)SpeciesNum*pao_stride], \
            pao_rv_flat[0:(size_t)SpeciesNum*pao_stride], den2_flat[0:(size_t)SpeciesNum*den_stride], \
            at_fnan[0:Matomnum+1], nb_off[0:Matomnum+1], \
            at_cx[0:Matomnum+1], at_cy[0:Matomnum+1], at_cz[0:Matomnum+1], at_dr[0:Matomnum+1], \
            nb_wan[0:nb_total], nb_x[0:nb_total], nb_y[0:nb_total], nb_z[0:nb_total], \
            nb_rcut2[0:nb_total], leb_x[0:na], leb_y[0:na], leb_z[0:na], leb_w[0:na], \
            CoarseGL_Abscissae[0:nr], CoarseGL_Weight[0:nr], pref[0:npt_total]) \
        map(from:item_fx[0:nitems_alloc], item_fy[0:nitems_alloc], item_fz[0:nitems_alloc])
      {
#pragma omp target teams distribute thread_limit(128)
        for (int pp=0; pp<nitems; pp++){
          const int mc = item_mc[pp];
          const int k = item_k[pp];
          const int koff = nb_off[mc];
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
          const size_t pt0 = (size_t)(mc-1)*(size_t)npt_atom;
          double sx = 0.0, sy = 0.0, sz = 0.0;

#pragma omp parallel for reduction(+:sx,sy,sz)
          for (int pt2=0; pt2<npt_atom; pt2++){
            int ir,ia;
            double r,x0,y0,z0,dx,dy,dz,r2;

            ir = pt2/na;
            ia = pt2 - ir*na;

            r = 0.50*(dr_atom*CoarseGL_Abscissae[ir] + dr_atom);
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
              gden0 = AMD_TotalEnergy_Exc0_Dr_KumoF_flat(pao_n, 0.5*log(r2), r1,
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

          item_fx[pp] = 2.0*PI*dr_atom*sx;
          item_fy[pp] = 2.0*PI*dr_atom*sy;
          item_fz[pp] = 2.0*PI*dr_atom*sz;
        }
      }

      /* host accumulation into the temporary force slots */

      p = 0;
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        for (h_AN=1; h_AN<=at_fnan[Mc_AN]; h_AN++){
          int Gh_AN = natn[Gc_AN][h_AN];

          Gxyz[Gh_AN][41] += item_fx[p];
          Gxyz[Gh_AN][42] += item_fy[p];
          Gxyz[Gh_AN][43] += item_fz[p];

          Gxyz[Gc_AN][41] -= item_fx[p];
          Gxyz[Gc_AN][42] -= item_fy[p];
          Gxyz[Gc_AN][43] -= item_fz[p];
          p++;
        }
      }

      free(item_mc);
      free(item_k);
      free(item_fx);
      free(item_fy);
      free(item_fz);
    }

  *sum_out = energy_sum;

  free(pao_n_flat);
  free(pao_xv_flat);
  free(pao_rv_flat);
  free(den2_flat);
  free(at_fnan);
  free(nb_off);
  free(at_cx);
  free(at_cy);
  free(at_cz);
  free(at_dr);
  free(nb_wan);
  free(nb_x);
  free(nb_y);
  free(nb_z);
  free(nb_rcut2);
  free(leb_x);
  free(leb_y);
  free(leb_z);
  free(leb_w);
  free(pref);
}



static int TotalEnergy_NativeAdmit(size_t required)
{
    const char *gpu=getenv("OPENMX_GPU"),*control=getenv("OPENMX_TOTALENERGY_GPU");
    const char *workers=getenv("OPENMX_GPU_WORKERS");
    int ok=scf_eigen_lib_flag==GPUSOLVER && !(gpu && atoi(gpu)==0) &&
        !(control && atoi(control)==0) && !openmx_gpu_capacity_test_recovery() &&
        !(workers && atoi(workers)==1);
    size_t free_bytes=0,total=0;
    if(ok){
        int sharing=openmx_gpu_local_size_noncollective();if(sharing<1)sharing=1;
        ok=gpu_rank_device_usable() && OpenMX_GpuMemGetInfo(&free_bytes,&total)==hipSuccess &&
            free_bytes>((size_t)512<<20) && required<=(free_bytes-((size_t)512<<20))/(size_t)sharing;
    }
    MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    return ok;
}

/* Host-backed grid jobs carry their geometry and controls to the helper
   node. No rank-local globals are read by the range kernel. */
typedef struct {
  int kind,spinmax,vna,field,exc0,n2,n3,GNs;
  double vectors[9],origin[3];
} TotalEnergyGridParams;

static int TotalEnergy_GPU_Enabled(void)
{
  const char *v=getenv("OPENMX_TOTALENERGY_GPU");
  return !v || !*v || atoi(v)!=0;
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
#pragma omp target teams distribute parallel for is_device_ptr(d0,d1,ad,pd0,pd1,dh,v0,v1,ref,vna,ef) reduction(+:s0,s1,s2,s3,s4) if(use_gpu)
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
#pragma omp target teams distribute parallel for is_device_ptr(d0,d1) reduction(+:s0,s1,s2,s3,s4,s5) if(use_gpu)
    for (size_t i=0;i<count;i++) {
      size_t grid=start+i,a=grid/(n2*n3),b=(grid-a*n2*n3)/n3,c=grid-a*n2*n3-b*n3;
      double x=a*x1+b*x2+c*x3+ox,y=a*y1+b*y2+c*y3+oy,z=a*z1+b*z2+c*z3+oz,den=d0[i]+d1[i];
      s0+=den*x;s1+=den*y;s2+=den*z;s3+=x;s4+=y;s5+=z;
    }
  }
  else {
#pragma omp target teams distribute parallel for is_device_ptr(d0,d1,ad,dh,v0,v1) reduction(+:s0,s1,s2,s3) if(use_gpu)
    for (size_t i=0;i<count;i++) {
      s0+=(d0[i]+ad[i])*dh[i];s2+=d0[i]*v0[i];
      if (spinmax>0) {s1+=(d1[i]+ad[i])*dh[i];s3+=d1[i]*v1[i];}
    }
  }
#pragma omp target is_device_ptr(out) if(use_gpu)
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

void TotalEnergy_EXC_EH1_Grid_OpenMP(int spinmax,double *My_Ena,double *My_Eef,double *My_EH1,double My_EXC[2])
{
  if(TotalEnergy_NativeAdmit((size_t)My_NumGridB_AB*11*sizeof(double))){AMD_TotalEnergy_EXC_EH1_Grid_OpenMP(spinmax,My_Ena,My_Eef,My_EH1,My_EXC);return;}

  double sums[6];TotalEnergy_Grid_Capacity(0,spinmax,0,sums);
  *My_Ena+=sums[0];*My_Eef+=sums[1];*My_EH1+=sums[2];My_EXC[0]+=sums[3];My_EXC[1]+=sums[4];
}

void TotalEnergy_Dipole_Grid_OpenMP(int GNs,double *My_E_dpx,double *My_E_dpy,double *My_E_dpz,
    double *My_E_dpx_BG,double *My_E_dpy_BG,double *My_E_dpz_BG)
{
  if(TotalEnergy_NativeAdmit((size_t)My_NumGridB_AB*11*sizeof(double))){AMD_TotalEnergy_Dipole_Grid_OpenMP(GNs,My_E_dpx,My_E_dpy,My_E_dpz,My_E_dpx_BG,My_E_dpy_BG,My_E_dpz_BG);return;}

  double sums[6];TotalEnergy_Grid_Capacity(1,0,GNs,sums);
  *My_E_dpx+=sums[0];*My_E_dpy+=sums[1];*My_E_dpz+=sums[2];
  *My_E_dpx_BG+=sums[3];*My_E_dpy_BG+=sums[4];*My_E_dpz_BG+=sums[5];
}

void TotalEnergy_CWF_Dc_Grid_OpenMP(int spinmax,double My_dcEH1[2],double My_dcEXC[2])
{
  if(TotalEnergy_NativeAdmit((size_t)My_NumGridB_AB*11*sizeof(double))){AMD_TotalEnergy_CWF_Dc_Grid_OpenMP(spinmax,My_dcEH1,My_dcEXC);return;}

  double sums[6];TotalEnergy_Grid_Capacity(2,spinmax,0,sums);
  for (int spin=0;spin<2;spin++) {My_dcEH1[spin]+=sums[spin];My_dcEXC[spin]+=sums[2+spin];}
}

#pragma omp declare target
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
#pragma omp end declare target


#pragma omp declare target
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
#pragma omp end declare target


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
#pragma omp target teams distribute parallel for is_device_ptr(pair_ban,pair_wan2,pair_has_deriv,pair_dis,pair_dirx,pair_diry,pair_dirz,tgn_flat,vps_n_flat,dv_flat,core_charge_flat,gridx_flat,gridy_flat,gridz_flat,arho_flat,wt_flat,vps_xv_flat,vps_rv_flat,vh_atom_flat,out0,out1,out2,out3) if(use_gpu)
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

void TotalEnergy_EH0_TwoCenter_Batch_OpenMP(int pair_count, int *pair_ban, int *pair_wan2,
                                             int *pair_has_deriv, double *pair_dis,
                                             double *pair_dirx, double *pair_diry, double *pair_dirz,
                                             double *out0, double *out1, double *out2, double *out3)
{
  if(TotalEnergy_NativeAdmit((size_t)SpeciesNum*((size_t)Max_TGN_EH0*5+(size_t)List_YOUSO[22]*3+8)*sizeof(double)+(size_t)pair_count*96)){AMD_TotalEnergy_EH0_TwoCenter_Batch_OpenMP(pair_count,pair_ban,pair_wan2,pair_has_deriv,pair_dis,pair_dirx,pair_diry,pair_dirz,out0,out1,out2,out3);return;}

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
    printf("TotalEnergy_EH0_TwoCenter_Batch_OpenMP: malloc failed.\n");
    fflush(stdout);
    exit(1);
  }

  for (spe=0; spe<species_count; spe++){
    tgn_flat[spe] = TGN_EH0[spe];
    vps_n_flat[spe] = Spe_Num_Mesh_VPS[spe];
    dv_flat[spe] = dv_EH0[spe];
    core_charge_flat[spe] = Spe_Core_Charge[spe];

    for (n=0; n<grid_stride; n++){
      i = spe*grid_stride + n;
      if (n<TGN_EH0[spe]){
        gridx_flat[i] = GridX_EH0[spe][n];
        gridy_flat[i] = GridY_EH0[spe][n];
        gridz_flat[i] = GridZ_EH0[spe][n];
        arho_flat[i] = Arho_EH0[spe][n];
        wt_flat[i] = Wt_EH0[spe][n];
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
      vps_xv_flat[i] = Spe_VPS_XV[spe][n];
      vps_rv_flat[i] = Spe_VPS_RV[spe][n];
    }

    for (n=0; n<vh_stride; n++){
      i = spe*vh_stride + n;
      vh_atom_flat[i] = Spe_VH_Atom[spe][n];
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


#pragma omp declare target
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
#pragma omp end declare target


#pragma omp declare target
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
#pragma omp end declare target


#pragma omp declare target
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
#pragma omp end declare target


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

#pragma omp target teams distribute parallel for is_device_ptr(at_fnan,at_cx,at_cy,at_cz,at_dr,nb_wan,nb_x,nb_y,nb_z,nb_rcut2,pao_n_flat,pao_xv_flat,pao_rv_flat,den2_flat,leb_x,leb_y,leb_z,leb_w,abscissa,weight,energy,force_pairs,pref) if(use_gpu)
  for (size_t mc=0;mc<count;mc++) {
    double energy_sum=0;

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
#pragma omp target teams distribute parallel for is_device_ptr(at_fnan,at_cx,at_cy,at_cz,at_dr,nb_wan,nb_x,nb_y,nb_z,nb_rcut2,pao_n_flat,pao_xv_flat,pao_rv_flat,den2_flat,leb_x,leb_y,leb_z,leb_w,abscissa,weight,energy,force_pairs,pref) if(use_gpu)
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

void TotalEnergy_Exc0_Batch_OpenMP(int na,double **leb,double *sum)
{
  if(TotalEnergy_NativeAdmit((size_t)Matomnum*CoarseGL_Mesh*na*64+(size_t)Matomnum*List_YOUSO[8]*128+(size_t)SpeciesNum*List_YOUSO[21]*24)){AMD_TotalEnergy_Exc0_Batch_OpenMP(na,leb,sum);return;}

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
    pao_n_flat[spe]=Spe_Num_Mesh_PAO[spe];
    memcpy(pao_xv_flat+(size_t)spe*p.pao_stride,Spe_PAO_XV[spe],(size_t)p.pao_stride*sizeof(double));
    memcpy(pao_rv_flat+(size_t)spe*p.pao_stride,Spe_PAO_RV[spe],(size_t)p.pao_stride*sizeof(double));
    memcpy(den2_flat+(size_t)spe*p.den_stride,Spe_Atomic_Den2[spe],(size_t)p.den_stride*sizeof(double));
  }
  for (int mc=0;mc<Matomnum;mc++) {
    int gc=M2G[mc+1];at_fnan[mc]=FNAN[gc];
    at_cx[mc]=Gxyz[gc][1];at_cy[mc]=Gxyz[gc][2];at_cz[mc]=Gxyz[gc][3];at_dr[mc]=Spe_Atom_Cut1[WhatSpecies[gc]];
    for (int h=0;h<=FNAN[gc];h++) {
      int gh=natn[gc][h],rn=ncn[gc][h],wan=WhatSpecies[gh];size_t k=(size_t)mc*nbmax+h;
      nb_wan[k]=wan;nb_x[k]=Gxyz[gh][1]+atv[rn][1];nb_y[k]=Gxyz[gh][2]+atv[rn][2];nb_z[k]=Gxyz[gh][3]+atv[rn][3];
      nb_rcut2[k]=Spe_Atom_Cut1[wan]*Spe_Atom_Cut1[wan];
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

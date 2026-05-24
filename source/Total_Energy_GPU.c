/**********************************************************************
  Total_Energy_GPU.c:

     OpenACC helper kernels for grid reductions in Total_Energy.c.

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"


void TotalEnergy_EXC_EH1_Grid_OpenACC(int spinmax, double *My_Ena, double *My_Eef,
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

#pragma acc data copyin(Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count], \
                        ADensity_Grid_B[0:grid_count], PCCDensity_Grid_B0[0:grid_count], \
                        PCCDensity_Grid_B1[0:grid_count], dVHart_Grid_B[0:grid_count], \
                        Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count], \
                        RefVxc_Grid_B[0:grid_count], VNA_Grid_B[0:vna_grid_count], \
                        VEF_Grid_B[0:vef_grid_count])
  {
#pragma acc parallel loop reduction(+:local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1)
    for (BN=0; BN<grid_count; BN++){
      double sden0,sden1,tden,aden,pden0,pden1,refvxc;

      sden0 = Density_Grid_B0[BN];
      sden1 = Density_Grid_B1[BN];
      tden = sden0 + sden1;
      aden = ADensity_Grid_B[BN];
      pden0 = PCCDensity_Grid_B0[BN];
      pden1 = PCCDensity_Grid_B1[BN];
      refvxc = RefVxc_Grid_B[BN];

      if (ProExpn_VNA==0) local_Ena += tden*VNA_Grid_B[BN];
      if (E_Field_switch==1) local_Eef += tden*VEF_Grid_B[BN];

      local_EH1 += (tden - 2.0*aden)*dVHart_Grid_B[BN];

      if (Exc0_correction_flag==1){
        local_EXC0 += (sden0+pden0)*Vxc_Grid_B0[BN] - (aden+pden0)*refvxc;
        if (0<spinmax) local_EXC1 += (sden1+pden1)*Vxc_Grid_B1[BN] - (aden+pden1)*refvxc;
      }
      else{
        local_EXC0 += (sden0+pden0)*Vxc_Grid_B0[BN];
        if (0<spinmax) local_EXC1 += (sden1+pden1)*Vxc_Grid_B1[BN];
      }
    }
  }

  *My_Ena = local_Ena;
  *My_Eef = local_Eef;
  *My_EH1 = local_EH1;
  My_EXC[0] = local_EXC0;
  My_EXC[1] = local_EXC1;
}


void TotalEnergy_Dipole_Grid_OpenACC(int GNs, double *My_E_dpx, double *My_E_dpy,
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

#pragma acc data copyin(Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count])
  {
#pragma acc parallel loop reduction(+:local_E_dpx,local_E_dpy,local_E_dpz,local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG)
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


void TotalEnergy_CWF_Dc_Grid_OpenACC(int spinmax, double My_dcEH1[2], double My_dcEXC[2])
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

#pragma acc data copyin(Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count], \
                        ADensity_Grid_B[0:grid_count], dVHart_Grid_B[0:grid_count], \
                        Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count])
  {
#pragma acc parallel loop reduction(+:local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1)
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


void TotalEnergy_EH0_TwoCenter_Batch_OpenACC(int pair_count, int *pair_ban, int *pair_wan2,
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
    printf("TotalEnergy_EH0_TwoCenter_Batch_OpenACC: malloc failed.\n");
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

#pragma acc data copyin(pair_ban[0:pair_count], pair_wan2[0:pair_count], \
                        pair_has_deriv[0:pair_count], pair_dis[0:pair_count], \
                        pair_dirx[0:pair_count], pair_diry[0:pair_count], pair_dirz[0:pair_count], \
                        tgn_flat[0:species_count], vps_n_flat[0:species_count], \
                        dv_flat[0:species_count], core_charge_flat[0:species_count], \
                        gridx_flat[0:grid_count], gridy_flat[0:grid_count], \
                        gridz_flat[0:grid_count], arho_flat[0:grid_count], wt_flat[0:grid_count], \
                        vps_xv_flat[0:vps_count], vps_rv_flat[0:vps_count], vh_atom_flat[0:vh_count]) \
                 copyout(out0[0:pair_count], out1[0:pair_count], out2[0:pair_count], out3[0:pair_count])
  {
#pragma acc parallel loop gang
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

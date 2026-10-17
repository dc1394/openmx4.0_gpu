/* CPU reference quadrature from OpenMX commit 43d05a07.
   Kept independent of the capacity packet and worker implementations. */
static void Reference_EXC_EH1_Grid_OpenMP_Range(int first, int grid_count, int use_gpu, int spinmax, double *My_Ena, double *My_Eef,
                                      double *My_EH1, double My_EXC[2])
{
  double *local_ADensity_Grid_B=ADensity_Grid_B?ADensity_Grid_B+first:NULL;
  double *local_dVHart_Grid_B=dVHart_Grid_B?dVHart_Grid_B+first:NULL;
  double *local_RefVxc_Grid_B=RefVxc_Grid_B?RefVxc_Grid_B+first:NULL;
  double *local_VNA_Grid_B=VNA_Grid_B?VNA_Grid_B+first:NULL;
  double *local_VEF_Grid_B=VEF_Grid_B?VEF_Grid_B+first:NULL;

  int BN,vna_grid_count,vef_grid_count;
  double local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1;
  double *Density_Grid_B0,*Density_Grid_B1;
  double *PCCDensity_Grid_B0,*PCCDensity_Grid_B1;
  double *Vxc_Grid_B0,*Vxc_Grid_B1;

  vna_grid_count = (ProExpn_VNA==0) ? grid_count : 0;
  vef_grid_count = (E_Field_switch==1) ? grid_count : 0;
  Density_Grid_B0 = Density_Grid_B[0] + first;
  Density_Grid_B1 = Density_Grid_B[1] + first;
  PCCDensity_Grid_B0 = PCCDensity_Grid_B[0] + first;
  PCCDensity_Grid_B1 = PCCDensity_Grid_B[1] + first;
  Vxc_Grid_B0 = Vxc_Grid_B[0] + first;
  Vxc_Grid_B1 = Vxc_Grid_B[1] + first;

  local_Ena = *My_Ena;
  local_Eef = *My_Eef;
  local_EH1 = *My_EH1;
  local_EXC0 = My_EXC[0];
  local_EXC1 = My_EXC[1];

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count],                          local_ADensity_Grid_B[0:grid_count], PCCDensity_Grid_B0[0:grid_count],                          PCCDensity_Grid_B1[0:grid_count], local_dVHart_Grid_B[0:grid_count],                          Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count],                          local_RefVxc_Grid_B[0:grid_count], local_VNA_Grid_B[0:vna_grid_count],                          local_VEF_Grid_B[0:vef_grid_count]) if(use_gpu)
  {
#pragma omp target teams distribute parallel for reduction(+:local_Ena,local_Eef,local_EH1,local_EXC0,local_EXC1) if(use_gpu)
    for (BN=0; BN<grid_count; BN++){
      double sden0,sden1,tden,aden,pden0,pden1,refvxc;

      sden0 = Density_Grid_B0[BN];
      sden1 = Density_Grid_B1[BN];
      tden = sden0 + sden1;
      aden = local_ADensity_Grid_B[BN];
      pden0 = PCCDensity_Grid_B0[BN];
      pden1 = PCCDensity_Grid_B1[BN];
      refvxc = local_RefVxc_Grid_B[BN];

      if (ProExpn_VNA==0) local_Ena += tden*local_VNA_Grid_B[BN];
      if (E_Field_switch==1) local_Eef += tden*local_VEF_Grid_B[BN];

      local_EH1 += (tden - 2.0*aden)*local_dVHart_Grid_B[BN];

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


static void Reference_Dipole_Grid_OpenMP_Range(int first, int grid_count, int use_gpu, int GNs, double *My_E_dpx, double *My_E_dpy,
                                     double *My_E_dpz, double *My_E_dpx_BG,
                                     double *My_E_dpy_BG, double *My_E_dpz_BG)
{
  int BN;
  double local_E_dpx,local_E_dpy,local_E_dpz;
  double local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG;
  double gtv11,gtv12,gtv13,gtv21,gtv22,gtv23,gtv31,gtv32,gtv33;
  double grid_origin1,grid_origin2,grid_origin3;
  double *Density_Grid_B0,*Density_Grid_B1;

  Density_Grid_B0 = Density_Grid_B[0] + first;
  Density_Grid_B1 = Density_Grid_B[1] + first;
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

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count]) if(use_gpu)
  {
#pragma omp target teams distribute parallel for reduction(+:local_E_dpx,local_E_dpy,local_E_dpz,local_E_dpx_BG,local_E_dpy_BG,local_E_dpz_BG) if(use_gpu)
    for (BN=0; BN<grid_count; BN++){
      int GN,n1,n2,n3;
      double x,y,z,den;

      GN = BN + GNs + first;
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


static void Reference_CWF_Dc_Grid_OpenMP_Range(int first, int grid_count, int use_gpu, int spinmax, double My_dcEH1[2], double My_dcEXC[2])
{
  double *local_ADensity_Grid_B=ADensity_Grid_B?ADensity_Grid_B+first:NULL;
  double *local_dVHart_Grid_B=dVHart_Grid_B?dVHart_Grid_B+first:NULL;

  int BN;
  double local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1;
  double *Density_Grid_B0,*Density_Grid_B1;
  double *Vxc_Grid_B0,*Vxc_Grid_B1;

  Density_Grid_B0 = Density_Grid_B[0] + first;
  Density_Grid_B1 = Density_Grid_B[1] + first;
  Vxc_Grid_B0 = Vxc_Grid_B[0] + first;
  Vxc_Grid_B1 = Vxc_Grid_B[1] + first;

  local_dcEH10 = My_dcEH1[0];
  local_dcEH11 = My_dcEH1[1];
  local_dcEXC0 = My_dcEXC[0];
  local_dcEXC1 = My_dcEXC[1];

#pragma omp target data map(to:Density_Grid_B0[0:grid_count], Density_Grid_B1[0:grid_count],                          local_ADensity_Grid_B[0:grid_count], local_dVHart_Grid_B[0:grid_count],                          Vxc_Grid_B0[0:grid_count], Vxc_Grid_B1[0:grid_count]) if(use_gpu)
  {
#pragma omp target teams distribute parallel for reduction(+:local_dcEH10,local_dcEH11,local_dcEXC0,local_dcEXC1) if(use_gpu)
    for (BN=0; BN<grid_count; BN++){
      double sden0,sden1,aden,dvhart;

      sden0 = Density_Grid_B0[BN];
      sden1 = Density_Grid_B1[BN];
      aden = local_ADensity_Grid_B[BN];
      dvhart = local_dVHart_Grid_B[BN];

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

static void Reference_EH0_TwoCenter(int use_gpu, int pair_count, int *pair_ban, int *pair_wan2,
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

#pragma omp target data map(to:pair_ban[0:pair_count], pair_wan2[0:pair_count],                          pair_has_deriv[0:pair_count], pair_dis[0:pair_count],                          pair_dirx[0:pair_count], pair_diry[0:pair_count], pair_dirz[0:pair_count],                          tgn_flat[0:species_count], vps_n_flat[0:species_count],                          dv_flat[0:species_count], core_charge_flat[0:species_count],                          gridx_flat[0:grid_count], gridy_flat[0:grid_count],                          gridz_flat[0:grid_count], arho_flat[0:grid_count], wt_flat[0:grid_count],                          vps_xv_flat[0:vps_count], vps_rv_flat[0:vps_count], vh_atom_flat[0:vh_count]) map(from:out0[0:pair_count], out1[0:pair_count], out2[0:pair_count], out3[0:pair_count]) if(use_gpu)
  {
#pragma omp target teams distribute parallel for if(use_gpu)
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

static void Reference_Exc0(int use_gpu,int Matomnum,const int *local_to_global,double *force_pairs,int Num_Leb, double **Leb_Grid_XYZW, double *sum_out)
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
    nb_total += FNAN[local_to_global[Mc_AN]] + 1;
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
    printf("TotalEnergy_Exc0_Batch_OpenMP: malloc failed.\n");
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

    Gc_AN = local_to_global[Mc_AN];
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

#pragma omp target data map(to:pao_n_flat[0:SpeciesNum],                          pao_xv_flat[0:(size_t)SpeciesNum*pao_stride],                          pao_rv_flat[0:(size_t)SpeciesNum*pao_stride],                          den2_flat[0:(size_t)SpeciesNum*den_stride],                          at_fnan[0:Matomnum+1], nb_off[0:Matomnum+1],                          at_cx[0:Matomnum+1], at_cy[0:Matomnum+1],                          at_cz[0:Matomnum+1], at_dr[0:Matomnum+1],                          nb_wan[0:nb_total], nb_x[0:nb_total], nb_y[0:nb_total],                          nb_z[0:nb_total], nb_rcut2[0:nb_total],                          leb_x[0:na], leb_y[0:na], leb_z[0:na], leb_w[0:na],                          CoarseGL_Abscissae[0:nr], CoarseGL_Weight[0:nr]) map(alloc:pref[0:npt_total]) if(use_gpu)
  {

    /* pass 1: energy and per-point force prefactors */

#pragma omp target teams distribute parallel for thread_limit(128) reduction(+:energy_sum) if(use_gpu)
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

      wpt = leb_w[ia]*r*r*CoarseGL_Weight[ir];

      pref[pt] = wpt*den0*dexc0;
      energy_sum += 2.0*PI*dr_atom*wpt*den0*exc0;
    }

    /* pass 2: contract the prefactors with the neighbour gradients */

    {
      int nitems = nb_total - Matomnum; /* h_AN != 0 entries */
      int *item_mc,*item_k;
      double *item_fx,*item_fy,*item_fz;
      int p;

      item_mc = (int*)malloc(sizeof(int)*(nitems==0 ? 1 : nitems));
      item_k = (int*)malloc(sizeof(int)*(nitems==0 ? 1 : nitems));
      item_fx = (double*)malloc(sizeof(double)*(nitems==0 ? 1 : nitems));
      item_fy = (double*)malloc(sizeof(double)*(nitems==0 ? 1 : nitems));
      item_fz = (double*)malloc(sizeof(double)*(nitems==0 ? 1 : nitems));

      if (item_mc==NULL || item_k==NULL || item_fx==NULL || item_fy==NULL || item_fz==NULL){
        printf("TotalEnergy_Exc0_Batch_OpenMP: malloc failed (pass 2).\n");
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

#pragma omp target data map(to:item_mc[0:(nitems==0 ? 1 : nitems)], item_k[0:(nitems==0 ? 1 : nitems)]) map(from:item_fx[0:(nitems==0 ? 1 : nitems)], item_fy[0:(nitems==0 ? 1 : nitems)],                           item_fz[0:(nitems==0 ? 1 : nitems)]) if(use_gpu)
      {
#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pao_n_flat[0:SpeciesNum],              pao_xv_flat[0:(size_t)SpeciesNum*pao_stride],              pao_rv_flat[0:(size_t)SpeciesNum*pao_stride],              den2_flat[0:(size_t)SpeciesNum*den_stride],              at_fnan[0:Matomnum+1], nb_off[0:Matomnum+1],              at_cx[0:Matomnum+1], at_cy[0:Matomnum+1],              at_cz[0:Matomnum+1], at_dr[0:Matomnum+1],              nb_wan[0:nb_total], nb_x[0:nb_total], nb_y[0:nb_total],              nb_z[0:nb_total], nb_rcut2[0:nb_total],              leb_x[0:na], leb_y[0:na], leb_z[0:na], leb_w[0:na],              CoarseGL_Abscissae[0:nr], CoarseGL_Weight[0:nr],              pref[0:npt_total]) if(use_gpu)
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

          item_fx[pp] = 2.0*PI*dr_atom*sx;
          item_fy[pp] = 2.0*PI*dr_atom*sy;
          item_fz[pp] = 2.0*PI*dr_atom*sz;
        }
      }

      /* Pair results are private until every GPU has completed both passes. */
      for (p=0;p<nitems;p++) {
        force_pairs[3*p]=item_fx[p];
        force_pairs[3*p+1]=item_fy[p];
        force_pairs[3*p+2]=item_fz[p];
      }

      free(item_mc);
      free(item_k);
      free(item_fx);
      free(item_fy);
      free(item_fz);
    }
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

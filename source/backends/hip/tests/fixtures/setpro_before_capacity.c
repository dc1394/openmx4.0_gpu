/* Frozen GPU arithmetic from 43d05a07, used by the capacity regression. */
/* DSVNA */
      /* K0: spherical Bessel tables of every (pair, GL point) */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_r[0:CHUNK], pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum],              GL_NormK[0:GL_Mesh], sphb[0:sphb_len], sphbp[0:sphb_len])
      for (int pt=0; pt<np*GL_Mesh; pt++){
        const int pp2 = pt/GL_Mesh;
        const int i = pt - pp2*GL_Mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*GL_Mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*GL_Mesh + i] = tbp[LL];
        }
      }

      /* K1: Gauss-Legendre quadrature sums */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], pr_hwan[0:CHUNK], pr_h0[0:CHUNK], sp_lfi[0:SpeciesNum],              sp_ncmb[0:SpeciesNum], pro00[0:pro_len], pro01[0:pro_len],              vnab[0:vnab_len], sphb[0:sphb_len], sphbp[0:sphb_len],              sum0[0:sum_len], sumr0[0:sum_len])
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*num_rvna; q++){
        const int Lq = (int)(q % num_rvna);
        const int cq = (int)((q/num_rvna) % cmb_max);
        const int LLq = (int)((q/((size_t)num_rvna*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)num_rvna*cmb_max*nLL));
        const int spe = pr_spe[pp2];

        if (LLq<=sp_lfi[spe] && cq<sp_ncmb[spe]){
          const double *p00 = pro00 + ((size_t)spe*cmb_max+cq)*GL_Mesh;
          const double *p01 = pro01 + ((size_t)spe*cmb_max+cq)*GL_Mesh;
          const double *vb = vnab + ((size_t)pr_hwan[pp2]*num_rvna+Lq)*GL_Mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          double tmp1 = 0.0,tmp2 = 0.0;
          int i;

          for (i=0; i<GL_Mesh; i++){
            tmp1 += (p00[i]*sb[i])*vb[i];
            tmp2 += (p01[i]*sbp[i])*vb[i];
          }

          sum0[q] = tmp1;
          sumr0[q] = pr_h0[pp2] ? 0.0 : tmp2;
        }
        else{
          sum0[q] = 0.0;
          sumr0[q] = 0.0;
        }
      }

      /* K2: assemble TmpNL (the LL,m sums with Gaunt x Y_lm weights) */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum],              cmb_L0[0:SpeciesNum*cmb_max], vna_L1[0:num_rvna],              gnt[0:gnt_len], sh_tab[0:(size_t)CHUNK*nsh*6],              sum0[0:sum_len], sumr0[0:sum_len], tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){
          const int lfi = sp_lfi[spe];
          int LL,m;

          for (LL=0; LL<=lfi; LL++){

            const int Ls = -L0 + L1 + LL;
            double pw_r,pw_i;

            /* Im_pow(-1,Ls) with the original C modulo semantics */
            if (Ls%2==0){
              if (Ls%4==0){ pw_r = 1.0; pw_i = 0.0; }
              else        { pw_r = -1.0; pw_i = 0.0; }
            }
            else{
              if ((Ls+1)%4==0){ pw_r = 0.0; pw_i = 1.0; }
              else            { pw_r = 0.0; pw_i = -1.0; }
            }

            if ( abs(L1-LL)<=L0 && L0<=(L1+LL) ){

              m = M0 - M1;
              if (-LL<=m && m<=LL){

                const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
                const double cy_r = sh_tab[shoff+0];
                const double cy_i = sh_tab[shoff+1];
                const double cyt_r = sh_tab[shoff+2];
                const double cyt_i = sh_tab[shoff+3];
                const double cyp_r = sh_tab[shoff+4];
                const double cyp_i = sh_tab[shoff+5];
                const double cy1_r = pw_r*cy_r - pw_i*cy_i;
                const double cy1_i = pw_r*cy_i + pw_i*cy_r;
                const double cyt1_r = pw_r*cyt_r - pw_i*cyt_i;
                const double cyt1_i = pw_r*cyt_i + pw_i*cyt_r;
                const double cyp1_r = pw_r*cyp_r - pw_i*cyp_i;
                const double cyp1_i = pw_r*cyp_i + pw_i*cyp_r;
                const size_t gidx = ((((((size_t)L0*nM0 + (M0+l0max_all))*(l1max+1) + L1)*nM1
                                       + (M1+l1max))*nLL + LL)*nm + (m+lfi_max));
                const double gant = gnt[gidx];
                const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + cq)*num_rvna + Lq;
                const double tmp2 = gant*sum0[sq];
                const double tmp0 = gant*sumr0[sq];

                s_r += cy1_r*tmp2;
                s_i += cy1_i*tmp2;
                r_r += cy1_r*tmp0;
                r_i += cy1_i*tmp0;
                t_r += cyt1_r*tmp2;
                t_i += cyt1_i*tmp2;
                p_r += cyp1_r*tmp2;
                p_i += cyp1_i*tmp2;
              }
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K3: complex-to-real transforms and the DS_VNA elements */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], pr_h0[0:CHUNK], pr_r[0:CHUNK], pr_siT[0:CHUNK],              pr_coT[0:CHUNK], pr_siP[0:CHUNK], pr_coP[0:CHUNK], pr_tno[0:CHUNK],              sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max],              cmb_num0[0:SpeciesNum*cmb_max], vna_L1[0:num_rvna], vna_num1[0:num_rvna],              c2r[0:c2r_len], c2r_off[0:lc_max+2], tmpnl[0:tmpnl_len],              outbuf[0:out_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM1);
        const int Lq = (int)((e/nM1) % num_rvna);
        const int M0i = (int)((e/((size_t)nM1*num_rvna)) % nM0);
        const int cq = (int)((e/((size_t)nM1*num_rvna*nM0)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (cq<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cq] : -1;
        const int L1 = vna_L1[Lq];
        const int M0 = M0i - l0max_all;
        const int M1 = M1i - l1max;

        if (L0>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            /* first transform: Cmat(M0,k1) = sum_k0 conj(C2R0)*TmpNL(k0,k1) */

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + cq)*nM0 + (k0+l0max_all))
                                 *num_rvna + Lq)*nM1 + (k1+l1max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            /* second transform */

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+cq];
            const int num1 = vna_num1[Lq];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = ((size_t)pp2*4)*(size_t)mn_max;
            const size_t oidx = (size_t)row*num_proj + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*(float)s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(float)(siT*coP*sr_r + coT*coP/r*st_r
                               - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(float)(siT*siP*sr_r + coT*siP/r*st_r
                               + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(float)(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0f;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0f;
            }
          }
        }
      }


/* VNA23 */
      /* K0: spherical Bessel tables */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_r[0:CHUNK], pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum],              GL_NormK[0:GL_Mesh], sphb[0:sphb_len], sphbp[0:sphb_len])
      for (int pt=0; pt<np*GL_Mesh; pt++){
        const int pp2 = pt/GL_Mesh;
        const int i = pt - pp2*GL_Mesh;
        const int lfi = sp_lfi[pr_spe[pp2]];
        double tb[SETPRO_BES_LMAX];
        double tbp[SETPRO_BES_LMAX];
        int LL;

        SetPro_Spherical_Bessel2_dev(GL_NormK[i]*pr_r[pp2],lfi,tb,tbp);
        for (LL=0; LL<=lfi; LL++){
          sphb [((size_t)pp2*nLL + LL)*GL_Mesh + i] = tb[LL];
          sphbp[((size_t)pp2*nLL + LL)*GL_Mesh + i] = tbp[LL];
        }
      }

      /* K1: quadrature sums for every (pair, LL, c0<=c1) block */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], pr_vspe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum],              cmb_L0[0:SpeciesNum*cmb_max], crude[0:crude_len], prod[0:prod_len],              GL_NormK[0:GL_Mesh], GL_Weight[0:GL_Mesh],              sphb[0:sphb_len], sphbp[0:sphb_len], sum0[0:sum_len], sumr0[0:sum_len])
      for (size_t q=0; q<(size_t)np*nLL*cmb_max*cmb_max; q++){
        const int cb = (int)(q % cmb_max);
        const int ca = (int)((q/cmb_max) % cmb_max);
        const int LLq = (int)((q/((size_t)cmb_max*cmb_max)) % nLL);
        const int pp2 = (int)(q/((size_t)cmb_max*cmb_max*nLL));
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        double s = 0.0,sr = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 && LLq<=2*L1 &&
            abs(L1-LLq)<=L0 && L0<=(L1+LLq)){

          const double *cru = crude + (size_t)pr_vspe[pp2]*GL_Mesh;
          const double *pr2 = prod + ((((size_t)spe*cmb_max+ca)*cmb_max+cb)*nLL+LLq)*GL_Mesh;
          const double *sb = sphb + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          const double *sbp = sphbp + ((size_t)pp2*nLL + LLq)*GL_Mesh;
          int i;

          for (i=0; i<GL_Mesh; i++){
            const double Normk = GL_NormK[i];
            const double tmp10 = 0.50*Dk*cru[i]*pr2[i]*GL_Weight[i]*Normk*Normk;

            s += tmp10*sb[i];
            sr += tmp10*sbp[i]*Normk;
          }
        }

        sum0[q] = s;
        sumr0[q] = sr;
      }

      /* K2a: assemble the upper (L0<=L1) TmpHNA elements */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], sp_lfi[0:SpeciesNum], sp_ncmb[0:SpeciesNum],              cmb_L0[0:SpeciesNum*cmb_max], gnt2[0:gnt2_len],              sh_tab[0:(size_t)CHUNK*nsh*6], sum0[0:sum_len], sumr0[0:sum_len],              tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;
        double s_r = 0.0,s_i = 0.0;
        double r_r = 0.0,r_i = 0.0;
        double t_r = 0.0,t_i = 0.0;
        double p_r = 0.0,p_i = 0.0;

        if (L0>=0 && L1>=0 && L0<=L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const int m = M1 - M0;
          int LL;

          for (LL=0; LL<=2*L1; LL++){

            if (abs(L1-LL)<=L0 && L0<=(L1+LL) && -LL<=m && m<=LL){

              const size_t shoff = ((size_t)pp2*nsh + (size_t)LL*LL + (m+LL))*6;
              const double cy_r = sh_tab[shoff+0];
              const double cy_i = sh_tab[shoff+1];
              const double cyt_r = sh_tab[shoff+2];
              const double cyt_i = sh_tab[shoff+3];
              const double cyp_r = sh_tab[shoff+4];
              const double cyp_i = sh_tab[shoff+5];
              const size_t gidx = (((((size_t)L0*nM + (M0+l0max))*(l0max+1) + L1)*nM
                                    + (M1+l0max))*nLL + LL);
              const double gant = gnt2[gidx];
              const size_t sq = (((size_t)pp2*nLL + LL)*cmb_max + ca)*cmb_max + cb;
              const double sv = sum0[sq];
              const double srv = sumr0[sq];
              const double cs_r = cy_r*gant;
              const double cs_i = cy_i*gant;
              const double cst_r = cyt_r*gant;
              const double cst_i = cyt_i*gant;
              const double csp_r = cyp_r*gant;
              const double csp_i = cyp_i*gant;

              s_r += cs_r*sv;
              s_i += cs_i*sv;
              r_r += cs_r*srv;
              r_i += cs_i*srv;
              t_r += cst_r*sv;
              t_i += cst_i*sv;
              p_r += csp_r*sv;
              p_i += csp_i*sv;
            }
          }
        }

        {
          double *dst = tmpnl + e*8;

          dst[0] = s_r; dst[1] = s_i;
          dst[2] = r_r; dst[3] = r_i;
          dst[4] = t_r; dst[5] = t_i;
          dst[6] = p_r; dst[7] = p_i;
        }
      }

      /* K2b: conjugate copy of the upper part to the lower part */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max],              tmpnl[0:tmpnl_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && L0>L1 &&
            M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          const size_t src = ((((size_t)pp2*cmb_max + cb)*nM + M1i)
                              *cmb_max + ca)*nM + M0i;
          const double *sv = tmpnl + src*8;
          double *dst = tmpnl + e*8;

          dst[0] = sv[0]; dst[1] = -sv[1];
          dst[2] = sv[2]; dst[3] = -sv[3];
          dst[4] = sv[4]; dst[5] = -sv[5];
          dst[6] = sv[6]; dst[7] = -sv[7];
        }
        else if (L0>=0 && L1>=0 && L0==L1 && ca==cb && M0==M1 &&
                 M0>=-L0 && M0<=L0){

          /* the host copy loop conjugates the L0==L1 diagonal in place */
          double *dst = tmpnl + e*8;

          dst[1] = -dst[1];
          dst[3] = -dst[3];
          dst[5] = -dst[5];
          dst[7] = -dst[7];
        }
      }

      /* K3: complex-to-real transforms and the HVNA2/3 elements */

#pragma omp target teams distribute parallel for thread_limit(128) map(present, alloc:pr_spe[0:CHUNK], pr_h0[0:CHUNK], pr_r[0:CHUNK], pr_siT[0:CHUNK],              pr_coT[0:CHUNK], pr_siP[0:CHUNK], pr_coP[0:CHUNK],              sp_ncmb[0:SpeciesNum], cmb_L0[0:SpeciesNum*cmb_max],              cmb_num0[0:SpeciesNum*cmb_max], c2r[0:c2r_len], c2r_off[0:l0max+2],              tmpnl[0:tmpnl_len], outbuf[0:out_len])
      for (size_t e=0; e<(size_t)np*elem_per_pair; e++){
        const int M1i = (int)(e % nM);
        const int cb = (int)((e/nM) % cmb_max);
        const int M0i = (int)((e/((size_t)nM*cmb_max)) % nM);
        const int ca = (int)((e/((size_t)nM*cmb_max*nM)) % cmb_max);
        const int pp2 = (int)(e/elem_per_pair);
        const int spe = pr_spe[pp2];
        const int L0 = (ca<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+ca] : -1;
        const int L1 = (cb<sp_ncmb[spe]) ? cmb_L0[spe*cmb_max+cb] : -1;
        const int M0 = M0i - l0max;
        const int M1 = M1i - l0max;

        if (L0>=0 && L1>=0 && M0>=-L0 && M0<=L0 && M1>=-L1 && M1<=L1){

          double s0_r = 0.0,s0_i = 0.0;
          double sr_r = 0.0,sr_i = 0.0;
          double st_r = 0.0,st_i = 0.0;
          double sp_r = 0.0,sp_i = 0.0;
          int k1,k0;

          for (k1=-L1; k1<=L1; k1++){

            double c0_r = 0.0,c0_i = 0.0;
            double cr_r = 0.0,cr_i = 0.0;
            double ct_r = 0.0,ct_i = 0.0;
            double cp_r = 0.0,cp_i = 0.0;

            for (k0=-L0; k0<=L0; k0++){
              const size_t uidx = (size_t)c2r_off[L0] + (size_t)(L0+M0)*(2*L0+1) + (L0+k0);
              const double u_r = c2r[2*uidx];
              const double u_i = -c2r[2*uidx+1];
              const size_t te = ((((size_t)pp2*cmb_max + ca)*nM + (k0+l0max))
                                 *cmb_max + cb)*nM + (k1+l0max);
              const double *tv = tmpnl + te*8;

              c0_r += u_r*tv[0] - u_i*tv[1];
              c0_i += u_r*tv[1] + u_i*tv[0];
              cr_r += u_r*tv[2] - u_i*tv[3];
              cr_i += u_r*tv[3] + u_i*tv[2];
              ct_r += u_r*tv[4] - u_i*tv[5];
              ct_i += u_r*tv[5] + u_i*tv[4];
              cp_r += u_r*tv[6] - u_i*tv[7];
              cp_i += u_r*tv[7] + u_i*tv[6];
            }

            {
              const size_t vidx = (size_t)c2r_off[L1] + (size_t)(L1+M1)*(2*L1+1) + (L1+k1);
              const double v_r = c2r[2*vidx];
              const double v_i = c2r[2*vidx+1];

              s0_r += (c0_r*v_r - c0_i*v_i);
              s0_i += (c0_r*v_i + c0_i*v_r);
              sr_r += (cr_r*v_r - cr_i*v_i);
              sr_i += (cr_r*v_i + cr_i*v_r);
              st_r += (ct_r*v_r - ct_i*v_i);
              st_i += (ct_r*v_i + ct_i*v_r);
              sp_r += (cp_r*v_r - cp_i*v_i);
              sp_i += (cp_r*v_i + cp_i*v_r);
            }
          }

          {
            const int num0 = cmb_num0[spe*cmb_max+ca];
            const int num1 = cmb_num0[spe*cmb_max+cb];
            const int row = num0 + L0 + M0;
            const int col = num1 + L1 + M1;
            const size_t obase = (size_t)pp2*4*(size_t)mn_max;
            const size_t oidx = (size_t)row*tno_max + col;
            const double r = pr_r[pp2];
            const double siT = pr_siT[pp2];
            const double coT = pr_coT[pp2];
            const double siP = pr_siP[pp2];
            const double coP = pr_coP[pp2];

            outbuf[obase + oidx] = 8.0*s0_r;

            if (!pr_h0[pp2]){

              if (fabs(siT)<10e-14){
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
              else{
                outbuf[obase + (size_t)mn_max + oidx] =
                  -8.0*(siT*coP*sr_r + coT*coP/r*st_r
                        - siP/siT/r*sp_r);
                outbuf[obase + 2*(size_t)mn_max + oidx] =
                  -8.0*(siT*siP*sr_r + coT*siP/r*st_r
                        + coP/siT/r*sp_r);
                outbuf[obase + 3*(size_t)mn_max + oidx] =
                  -8.0*(coT*sr_r - siT/r*st_r);
              }
            }
            else{
              outbuf[obase + (size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 2*(size_t)mn_max + oidx] = 0.0;
              outbuf[obase + 3*(size_t)mn_max + oidx] = 0.0;
            }
          }
        }
      }

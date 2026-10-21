/* Frozen host Hubbard derivatives from 6e7a8dad, for numerical regression. */
void reference_dH_U_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, double**** v_eff,
    double*** Hx, double*** Hy, double*** Hz, const OpenMXLocalBasisViews *local_basis)
{

    int i, j, k, m, n, kg, kan, so, deri_kind, Mk_AN;
    int ig, ian, jg, jan, kl, kl1, kl2, spin, spinmax;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mj_AN;
    int Rwan, Lwan, p, p0;
    double PF[2], sumx, sumy, sumz, ene;
    double tmpx, tmpy, tmpz;
    double Lsum0, Lsum1, Lsum2, Lsum3;
    double Rsum0, Rsum1, Rsum2, Rsum3;
    double PFp, PFm, ene_p, ene_m;
    double ***Hx2, ***Hy2, ***Hz2;
    double sumx0, sumy0, sumz0;
    double sumx1, sumy1, sumz1;
    double sumx2, sumy2, sumz2;

    /****************************************************
     allocation of arrays:

     double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
    ****************************************************/

    Hx2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hx2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hx2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    Hy2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hy2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hy2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    Hz2 = (double***)malloc(sizeof(double**) * 3);
    for (i = 0; i < 3; i++) {
        Hz2[i] = (double**)malloc(sizeof(double*) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++) {
            Hz2[i][j] = (double*)malloc(sizeof(double) * List_YOUSO[7]);
        }
    }

    /****************************************************
     start calc.
    ****************************************************/

    if (SpinP_switch == 0)
        spinmax = 0;
    else
        spinmax = 1;

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Lwan = WhatSpecies[ig];
    Mi_AN = F_G2M[ig]; /* F_G2M should be used */
    ian = Spe_Total_CNO[Lwan];
    jg = natn[Gc_AN][q_AN];
    Rwan = WhatSpecies[jg];
    Mj_AN = F_G2M[jg]; /* F_G2M should be used */
    jan = Spe_Total_CNO[Rwan];

    if (h_AN == 0) {

        /****************************************************
                              dS*ep*S
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                        derivative at h_AN (=Mc_AN)
            ****************************************************/

            if (0 <= kl) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        for (spin = 0; spin <= spinmax; spin++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

                            if (Cnt_switch == 0) {

                                for (l1 = 0; l1 < kan; l1++) {
                                    for (l2 = 0; l2 < kan; l2++) {
                                        ene = v_eff[spin][Mk_AN][l1][l2];
                                        sumx += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                        sumy += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                        sumz += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                    }
                                }

                            }

                            else if (Cnt_switch == 1) {

                                for (l1 = 0; l1 < kan; l1++) {
                                    for (l2 = 0; l2 < kan; l2++) {
                                        Lsum1 = 0.0;
                                        Lsum2 = 0.0;
                                        Lsum3 = 0.0;
                                        for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                            p0 = Spe_Trans_Orbital[Lwan][m][p];
                                            Lsum1 += reference_contraction_row(local_basis, Mc_AN, m)[p] * OLP[1][Mc_AN][k][p0][l1];
                                            Lsum2 += reference_contraction_row(local_basis, Mc_AN, m)[p] * OLP[2][Mc_AN][k][p0][l1];
                                            Lsum3 += reference_contraction_row(local_basis, Mc_AN, m)[p] * OLP[3][Mc_AN][k][p0][l1];
                                        }

                                        Rsum0 = 0.0;
                                        for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                            p0 = Spe_Trans_Orbital[Rwan][n][p];
                                            Rsum0 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[0][Mj_AN][kl][p0][l2];
                                        }

                                        ene = v_eff[spin][Mk_AN][l1][l2];
                                        sumx += ene * Lsum1 * Rsum0;
                                        sumy += ene * Lsum2 * Rsum0;
                                        sumz += ene * Lsum3 * Rsum0;
                                    }
                                }
                            }

                            if (k == 0) {
                                Hx[spin][m][n] = sumx;
                                Hy[spin][m][n] = sumy;
                                Hz[spin][m][n] = sumz;

                                Hx[2][m][n] = 0.0;
                                Hy[2][m][n] = 0.0;
                                Hz[2][m][n] = 0.0;
                            } else {
                                Hx[spin][m][n] += sumx;
                                Hy[spin][m][n] += sumy;
                                Hz[spin][m][n] += sumz;
                            }
                        }
                    }
                }
            } /* if */
        } /* k */

        /****************************************************
                              S*ep*dS
        ****************************************************/

        if (q_AN == 0) {
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx2[0][m][n] = Hx[0][m][n];
                    Hy2[0][m][n] = Hy[0][m][n];
                    Hz2[0][m][n] = Hz[0][m][n];

                    Hx2[1][m][n] = Hx[1][m][n];
                    Hy2[1][m][n] = Hy[1][m][n];
                    Hz2[1][m][n] = Hz[1][m][n];
                }
            }
            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {
                    Hx[0][m][n] = Hx2[0][m][n] + Hx2[0][n][m];
                    Hy[0][m][n] = Hy2[0][m][n] + Hy2[0][n][m];
                    Hz[0][m][n] = Hz2[0][m][n] + Hz2[0][n][m];

                    Hx[1][m][n] = Hx2[1][m][n] + Hx2[1][n][m];
                    Hy[1][m][n] = Hy2[1][m][n] + Hy2[1][n][m];
                    Hz[1][m][n] = Hz2[1][m][n] + Hz2[1][n][m];
                }
            }
        }

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                              derivative at k=0
            ****************************************************/

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    for (spin = 0; spin <= spinmax; spin++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Cnt_switch == 0) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {
                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                                    sumy -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                                    sumz -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];
                                }
                            }

                        }

                        else if (Cnt_switch == 1) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    Lsum0 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                        p0 = Spe_Trans_Orbital[Lwan][m][p];
                                        Lsum0 += reference_contraction_row(local_basis, Mc_AN, m)[p] * OLP[0][Mc_AN][0][p0][l1];
                                    }

                                    Rsum1 = 0.0;
                                    Rsum2 = 0.0;
                                    Rsum3 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                        p0 = Spe_Trans_Orbital[Rwan][n][p];
                                        Rsum1 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[1][Mj_AN][kl][p0][l2];
                                        Rsum2 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[2][Mj_AN][kl][p0][l2];
                                        Rsum3 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[3][Mj_AN][kl][p0][l2];
                                    }

                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * Lsum0 * Rsum1;
                                    sumy -= ene * Lsum0 * Rsum2;
                                    sumz -= ene * Lsum0 * Rsum3;
                                }
                            }
                        }

                        Hx[spin][m][n] += sumx;
                        Hy[spin][m][n] += sumy;
                        Hz[spin][m][n] += sumz;
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else {

        /****************************************************
                               dS*ep*S
        ****************************************************/

        kg = natn[Gc_AN][0];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][0];
        kl2 = RMI1[Mc_AN][q_AN][0];

        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                for (spin = 0; spin <= spinmax; spin++) {

                    sumx = 0.0;
                    sumy = 0.0;
                    sumz = 0.0;

                    if (Cnt_switch == 0) {

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {
                                ene = v_eff[spin][Mk_AN][l1][l2];
                                sumx -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                                sumy -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                                sumz -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                            }
                        }
                    }

                    else if (Cnt_switch == 1) {

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {

                                Lsum1 = 0.0;
                                Lsum2 = 0.0;
                                Lsum3 = 0.0;
                                for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                    p0 = Spe_Trans_Orbital[Lwan][m][p];
                                    Lsum1 += reference_contraction_row(local_basis, Mi_AN, m)[p] * OLP[1][Mi_AN][kl1][p0][l1];
                                    Lsum2 += reference_contraction_row(local_basis, Mi_AN, m)[p] * OLP[2][Mi_AN][kl1][p0][l1];
                                    Lsum3 += reference_contraction_row(local_basis, Mi_AN, m)[p] * OLP[3][Mi_AN][kl1][p0][l1];
                                }

                                Rsum0 = 0.0;
                                for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                    p0 = Spe_Trans_Orbital[Rwan][n][p];
                                    Rsum0 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[0][Mj_AN][kl2][p0][l2];
                                }

                                ene = v_eff[spin][Mk_AN][l1][l2];
                                sumx -= ene * Lsum1 * Rsum0;
                                sumy -= ene * Lsum2 * Rsum0;
                                sumz -= ene * Lsum3 * Rsum0;
                            }
                        }
                    }

                    Hx[spin][m][n] = sumx;
                    Hy[spin][m][n] = sumy;
                    Hz[spin][m][n] = sumz;

                    Hx[2][m][n] = 0.0;
                    Hy[2][m][n] = 0.0;
                    Hz[2][m][n] = 0.0;
                }
            }
        }

        /****************************************************
                               S*ep*dS
        ****************************************************/

        if (q_AN == 0) {

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                kg = natn[Gc_AN][k];
                Mk_AN = F_G2M[kg]; /* F_G2M should be used */
                wakg = WhatSpecies[kg];
                kan = Spe_Total_NO[wakg];
                kl1 = RMI1[Mc_AN][h_AN][k];
                kl2 = RMI1[Mc_AN][q_AN][k];

                if (0 <= kl1) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            for (spin = 0; spin <= spinmax; spin++) {

                                sumx = 0.0;
                                sumy = 0.0;
                                sumz = 0.0;

                                if (Cnt_switch == 0) {

                                    for (l1 = 0; l1 < kan; l1++) {
                                        for (l2 = 0; l2 < kan; l2++) {
                                            ene = v_eff[spin][Mk_AN][l1][l2];
                                            sumx += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                            sumy += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                            sumz += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                        }
                                    }

                                }

                                else if (Cnt_switch == 1) {

                                    for (l1 = 0; l1 < kan; l1++) {
                                        for (l2 = 0; l2 < kan; l2++) {

                                            Lsum0 = 0.0;

                                            for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                                p0 = Spe_Trans_Orbital[Lwan][m][p];
                                                Lsum0 += reference_contraction_row(local_basis, Mi_AN, m)[p] * OLP[0][Mi_AN][kl1][p0][l1];
                                            }

                                            Rsum1 = 0.0;
                                            Rsum2 = 0.0;
                                            Rsum3 = 0.0;

                                            for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                                p0 = Spe_Trans_Orbital[Rwan][n][p];
                                                Rsum1 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[1][Mj_AN][kl2][p0][l2];
                                                Rsum2 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[2][Mj_AN][kl2][p0][l2];
                                                Rsum3 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[3][Mj_AN][kl2][p0][l2];
                                            }

                                            ene = v_eff[spin][Mk_AN][l1][l2];
                                            sumx += ene * Lsum0 * Rsum1;
                                            sumy += ene * Lsum0 * Rsum2;
                                            sumz += ene * Lsum0 * Rsum3;
                                        }
                                    }
                                }

                                Hx[spin][m][n] += sumx;
                                Hy[spin][m][n] += sumy;
                                Hz[spin][m][n] += sumz;
                            }
                        }
                    }
                }
            }
        } /* if (q_AN==0) */

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl1 = RMI1[Mc_AN][h_AN][0];
            kl2 = RMI1[Mc_AN][q_AN][0];

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    for (spin = 0; spin <= spinmax; spin++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        if (Cnt_switch == 0) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {
                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    sumy -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    sumz -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                }
                            }
                        }

                        else if (Cnt_switch == 1) {

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    Lsum0 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++) {
                                        p0 = Spe_Trans_Orbital[Lwan][m][p];
                                        Lsum0 += reference_contraction_row(local_basis, Mi_AN, m)[p] * OLP[0][Mi_AN][kl1][p0][l1];
                                    }

                                    Rsum1 = 0.0;
                                    Rsum2 = 0.0;
                                    Rsum3 = 0.0;

                                    for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++) {
                                        p0 = Spe_Trans_Orbital[Rwan][n][p];
                                        Rsum1 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[1][Mj_AN][kl2][p0][l2];
                                        Rsum2 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[2][Mj_AN][kl2][p0][l2];
                                        Rsum3 += reference_contraction_row(local_basis, Mj_AN, n)[p] * OLP[3][Mj_AN][kl2][p0][l2];
                                    }

                                    ene = v_eff[spin][Mk_AN][l1][l2];
                                    sumx -= ene * Lsum0 * Rsum1;
                                    sumy -= ene * Lsum0 * Rsum2;
                                    sumz -= ene * Lsum0 * Rsum3;
                                }
                            }
                        }

                        Hx[spin][m][n] += sumx;
                        Hy[spin][m][n] += sumy;
                        Hz[spin][m][n] += sumz;
                    }
                }
            }
        }
    }

    /****************************************************
     freeing of arrays:

     double Hx2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hy2[3][List_YOUSO[7]][List_YOUSO[7]];
     double Hz2[3][List_YOUSO[7]][List_YOUSO[7]];
    ****************************************************/

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hx2[i][j]);
        }
        free(Hx2[i]);
    }
    free(Hx2);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hy2[i][j]);
        }
        free(Hy2[i]);
    }
    free(Hy2);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < List_YOUSO[7]; j++) {
            free(Hz2[i][j]);
        }
        free(Hz2[i]);
    }
    free(Hz2);
}
void reference_dH_U_NC_full(int Mc_AN, int h_AN, int q_AN,
    double***** OLP, dcomplex***** NC_v_eff,
    dcomplex**** Hx, dcomplex**** Hy, dcomplex**** Hz)
{
    int i, j, k, m, n, kg, kan, so, deri_kind, Mk_AN;
    int ig, ian, jg, jan, kl, kl1, kl2, spin;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mj_AN;
    int Rwan, Lwan, p, p0, s1, s2;
    double PF[2], sumx, sumy, sumz, ene;
    double tmpx, tmpy, tmpz;
    double Lsum0, Lsum1, Lsum2, Lsum3;
    double Rsum0, Rsum1, Rsum2, Rsum3;
    double PFp, PFm, ene_p, ene_m;
    double Re00x, Re00y, Re00z;
    double Re11x, Re11y, Re11z;
    double Re01x, Re01y, Re01z;
    double Re10x, Re10y, Re10z;
    double Im00x, Im00y, Im00z;
    double Im11x, Im11y, Im11z;
    double Im01x, Im01y, Im01z;
    double Im10x, Im10y, Im10z;

    /****************************************************
     start calc.
    ****************************************************/

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Lwan = WhatSpecies[ig];
    Mi_AN = F_G2M[ig]; /* F_G2M should be used */
    ian = Spe_Total_CNO[Lwan];
    jg = natn[Gc_AN][q_AN];
    Rwan = WhatSpecies[jg];
    Mj_AN = F_G2M[jg]; /* F_G2M should be used */
    jan = Spe_Total_CNO[Rwan];

    if (h_AN == 0) {

        /****************************************************
                              dS*ep*S
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                        derivative at h_AN (=Mc_AN)
            ****************************************************/

            if (0 <= kl) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        Re00x = 0.0;
                        Re00y = 0.0;
                        Re00z = 0.0;
                        Re11x = 0.0;
                        Re11y = 0.0;
                        Re11z = 0.0;
                        Re01x = 0.0;
                        Re01y = 0.0;
                        Re01z = 0.0;
                        Re10x = 0.0;
                        Re10y = 0.0;
                        Re10z = 0.0;

                        Im00x = 0.0;
                        Im00y = 0.0;
                        Im00z = 0.0;
                        Im11x = 0.0;
                        Im11y = 0.0;
                        Im11z = 0.0;
                        Im01x = 0.0;
                        Im01y = 0.0;
                        Im01z = 0.0;
                        Im10x = 0.0;
                        Im10y = 0.0;
                        Im10z = 0.0;

                        for (l1 = 0; l1 < kan; l1++) {
                            for (l2 = 0; l2 < kan; l2++) {

                                ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                                Re00x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re00y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re00z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                                Re11x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re11y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re11z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                                Re01x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re01y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re01z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                                Re10x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re10y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Re10z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                                Im00x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im00y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im00z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                                Im11x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im11y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im11z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                                Im01x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im01y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im01z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];

                                ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                                Im10x += ene * OLP[1][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im10y += ene * OLP[2][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                                Im10z += ene * OLP[3][Mc_AN][k][m][l1] * OLP[0][Mj_AN][kl][n][l2];
                            }
                        }

                        if (k == 0) {
                            Hx[0][0][m][n] = Complex(Re00x, Im00x);
                            Hy[0][0][m][n] = Complex(Re00y, Im00y);
                            Hz[0][0][m][n] = Complex(Re00z, Im00z);

                            Hx[1][1][m][n] = Complex(Re11x, Im11x);
                            Hy[1][1][m][n] = Complex(Re11y, Im11y);
                            Hz[1][1][m][n] = Complex(Re11z, Im11z);

                            Hx[0][1][m][n] = Complex(Re01x, Im01x);
                            Hy[0][1][m][n] = Complex(Re01y, Im01y);
                            Hz[0][1][m][n] = Complex(Re01z, Im01z);

                            Hx[1][0][m][n] = Complex(Re10x, Im10x);
                            Hy[1][0][m][n] = Complex(Re10y, Im10y);
                            Hz[1][0][m][n] = Complex(Re10z, Im10z);
                        } else {

                            Hx[0][0][m][n].r += Re00x;
                            Hx[0][0][m][n].i += Im00x;
                            Hy[0][0][m][n].r += Re00y;
                            Hy[0][0][m][n].i += Im00y;
                            Hz[0][0][m][n].r += Re00z;
                            Hz[0][0][m][n].i += Im00z;

                            Hx[1][1][m][n].r += Re11x;
                            Hx[1][1][m][n].i += Im11x;
                            Hy[1][1][m][n].r += Re11y;
                            Hy[1][1][m][n].i += Im11y;
                            Hz[1][1][m][n].r += Re11z;
                            Hz[1][1][m][n].i += Im11z;

                            Hx[0][1][m][n].r += Re01x;
                            Hx[0][1][m][n].i += Im01x;
                            Hy[0][1][m][n].r += Re01y;
                            Hy[0][1][m][n].i += Im01y;
                            Hz[0][1][m][n].r += Re01z;
                            Hz[0][1][m][n].i += Im01z;

                            Hx[1][0][m][n].r += Re10x;
                            Hx[1][0][m][n].i += Im10x;
                            Hy[1][0][m][n].r += Re10y;
                            Hy[1][0][m][n].i += Im10y;
                            Hz[1][0][m][n].r += Re10z;
                            Hz[1][0][m][n].i += Im10z;
                        }

                    } /* n */
                } /* m */
            } /* if */
        } /* k */

        /****************************************************
                              S*ep*dS
        ****************************************************/

        /* ????? */

        if (q_AN == 0) {

            for (s1 = 0; s1 < 2; s1++) {
                for (s2 = 0; s2 < 2; s2++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            Hx[s1][s2][m][n].r = 2.0 * Hx[s1][s2][m][n].r;
                            Hy[s1][s2][m][n].r = 2.0 * Hy[s1][s2][m][n].r;
                            Hz[s1][s2][m][n].r = 2.0 * Hz[s1][s2][m][n].r;

                            Hx[s1][s2][m][n].i = 2.0 * Hx[s1][s2][m][n].i;
                            Hy[s1][s2][m][n].i = 2.0 * Hy[s1][s2][m][n].i;
                            Hz[s1][s2][m][n].i = 2.0 * Hz[s1][s2][m][n].i;
                        }
                    }
                }
            }
        }

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                              derivative at k=0
            ****************************************************/

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    Re00x = 0.0;
                    Re00y = 0.0;
                    Re00z = 0.0;
                    Re11x = 0.0;
                    Re11y = 0.0;
                    Re11z = 0.0;
                    Re01x = 0.0;
                    Re01y = 0.0;
                    Re01z = 0.0;
                    Re10x = 0.0;
                    Re10y = 0.0;
                    Re10z = 0.0;

                    Im00x = 0.0;
                    Im00y = 0.0;
                    Im00z = 0.0;
                    Im11x = 0.0;
                    Im11y = 0.0;
                    Im11z = 0.0;
                    Im01x = 0.0;
                    Im01y = 0.0;
                    Im01z = 0.0;
                    Im10x = 0.0;
                    Im10y = 0.0;
                    Im10z = 0.0;

                    for (l1 = 0; l1 < kan; l1++) {
                        for (l2 = 0; l2 < kan; l2++) {

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                            Re00x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re00y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re00z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                            Re11x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re11y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re11z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                            Re01x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re01y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re01z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                            Re10x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Re10y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Re10z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                            Im00x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im00y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im00z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                            Im11x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im11y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im11z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                            Im01x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im01y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im01z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                            Im10x -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[1][Mj_AN][kl][n][l2];
                            Im10y -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[2][Mj_AN][kl][n][l2];
                            Im10z -= ene * OLP[0][Mc_AN][0][m][l1] * OLP[3][Mj_AN][kl][n][l2];
                        }
                    }

                    Hx[0][0][m][n].r += Re00x;
                    Hx[0][0][m][n].i += Im00x;
                    Hy[0][0][m][n].r += Re00y;
                    Hy[0][0][m][n].i += Im00y;
                    Hz[0][0][m][n].r += Re00z;
                    Hz[0][0][m][n].i += Im00z;

                    Hx[1][1][m][n].r += Re11x;
                    Hx[1][1][m][n].i += Im11x;
                    Hy[1][1][m][n].r += Re11y;
                    Hy[1][1][m][n].i += Im11y;
                    Hz[1][1][m][n].r += Re11z;
                    Hz[1][1][m][n].i += Im11z;

                    Hx[0][1][m][n].r += Re01x;
                    Hx[0][1][m][n].i += Im01x;
                    Hy[0][1][m][n].r += Re01y;
                    Hy[0][1][m][n].i += Im01y;
                    Hz[0][1][m][n].r += Re01z;
                    Hz[0][1][m][n].i += Im01z;

                    Hx[1][0][m][n].r += Re10x;
                    Hx[1][0][m][n].i += Im10x;
                    Hy[1][0][m][n].r += Re10y;
                    Hy[1][0][m][n].i += Im10y;
                    Hz[1][0][m][n].r += Re10z;
                    Hz[1][0][m][n].i += Im10z;
                }
            }
        }

    } /* if (h_AN==0) */

    else {

        /****************************************************
                               dS*ep*S
        ****************************************************/

        kg = natn[Gc_AN][0];
        Mk_AN = F_G2M[kg]; /* F_G2M should be used */
        wakg = WhatSpecies[kg];
        kan = Spe_Total_NO[wakg];
        kl1 = RMI1[Mc_AN][h_AN][0];
        kl2 = RMI1[Mc_AN][q_AN][0];

        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Re00x = 0.0;
                Re00y = 0.0;
                Re00z = 0.0;
                Re11x = 0.0;
                Re11y = 0.0;
                Re11z = 0.0;
                Re01x = 0.0;
                Re01y = 0.0;
                Re01z = 0.0;
                Re10x = 0.0;
                Re10y = 0.0;
                Re10z = 0.0;

                Im00x = 0.0;
                Im00y = 0.0;
                Im00z = 0.0;
                Im11x = 0.0;
                Im11y = 0.0;
                Im11z = 0.0;
                Im01x = 0.0;
                Im01y = 0.0;
                Im01z = 0.0;
                Im10x = 0.0;
                Im10y = 0.0;
                Im10z = 0.0;

                for (l1 = 0; l1 < kan; l1++) {
                    for (l2 = 0; l2 < kan; l2++) {

                        ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                        Re00x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re00y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re00z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                        Re11x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re11y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re11z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                        Re01x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re01y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re01z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                        Re10x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re10y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Re10z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                        Im00x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im00y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im00z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                        Im11x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im11y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im11z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                        Im01x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im01y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im01z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];

                        ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                        Im10x -= ene * OLP[1][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im10y -= ene * OLP[2][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                        Im10z -= ene * OLP[3][Mi_AN][kl1][m][l1] * OLP[0][Mj_AN][kl2][n][l2];
                    }
                }

                Hx[0][0][m][n] = Complex(Re00x, Im00x);
                Hy[0][0][m][n] = Complex(Re00y, Im00y);
                Hz[0][0][m][n] = Complex(Re00z, Im00z);

                Hx[1][1][m][n] = Complex(Re11x, Im11x);
                Hy[1][1][m][n] = Complex(Re11y, Im11y);
                Hz[1][1][m][n] = Complex(Re11z, Im11z);

                Hx[0][1][m][n] = Complex(Re01x, Im01x);
                Hy[0][1][m][n] = Complex(Re01y, Im01y);
                Hz[0][1][m][n] = Complex(Re01z, Im01z);

                Hx[1][0][m][n] = Complex(Re10x, Im10x);
                Hy[1][0][m][n] = Complex(Re10y, Im10y);
                Hz[1][0][m][n] = Complex(Re10z, Im10z);
            }
        }

        /****************************************************
                               S*ep*dS
        ****************************************************/

        if (q_AN == 0) {

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                kg = natn[Gc_AN][k];
                Mk_AN = F_G2M[kg]; /* F_G2M should be used */
                wakg = WhatSpecies[kg];
                kan = Spe_Total_NO[wakg];
                kl1 = RMI1[Mc_AN][h_AN][k];
                kl2 = RMI1[Mc_AN][q_AN][k];

                if (0 <= kl1) {

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            Re00x = 0.0;
                            Re00y = 0.0;
                            Re00z = 0.0;
                            Re11x = 0.0;
                            Re11y = 0.0;
                            Re11z = 0.0;
                            Re01x = 0.0;
                            Re01y = 0.0;
                            Re01z = 0.0;
                            Re10x = 0.0;
                            Re10y = 0.0;
                            Re10z = 0.0;

                            Im00x = 0.0;
                            Im00y = 0.0;
                            Im00z = 0.0;
                            Im11x = 0.0;
                            Im11y = 0.0;
                            Im11z = 0.0;
                            Im01x = 0.0;
                            Im01y = 0.0;
                            Im01z = 0.0;
                            Im10x = 0.0;
                            Im10y = 0.0;
                            Im10z = 0.0;

                            for (l1 = 0; l1 < kan; l1++) {
                                for (l2 = 0; l2 < kan; l2++) {

                                    ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                                    Re00x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re00y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re00z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                                    Re11x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re11y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re11z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                                    Re01x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re01y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re01z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                                    Re10x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Re10y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Re10z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                                    Im00x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im00y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im00z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                                    Im11x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im11y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im11z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                                    Im01x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im01y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im01z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                                    ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                                    Im10x += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                                    Im10y += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                                    Im10z += ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                                }
                            }

                            Hx[0][0][m][n].r += Re00x;
                            Hx[0][0][m][n].i += Im00x;
                            Hy[0][0][m][n].r += Re00y;
                            Hy[0][0][m][n].i += Im00y;
                            Hz[0][0][m][n].r += Re00z;
                            Hz[0][0][m][n].i += Im00z;

                            Hx[1][1][m][n].r += Re11x;
                            Hx[1][1][m][n].i += Im11x;
                            Hy[1][1][m][n].r += Re11y;
                            Hy[1][1][m][n].i += Im11y;
                            Hz[1][1][m][n].r += Re11z;
                            Hz[1][1][m][n].i += Im11z;

                            Hx[0][1][m][n].r += Re01x;
                            Hx[0][1][m][n].i += Im01x;
                            Hy[0][1][m][n].r += Re01y;
                            Hy[0][1][m][n].i += Im01y;
                            Hz[0][1][m][n].r += Re01z;
                            Hz[0][1][m][n].i += Im01z;

                            Hx[1][0][m][n].r += Re10x;
                            Hx[1][0][m][n].i += Im10x;
                            Hy[1][0][m][n].r += Re10y;
                            Hy[1][0][m][n].i += Im10y;
                            Hz[1][0][m][n].r += Re10z;
                            Hz[1][0][m][n].i += Im10z;
                        }
                    }
                }
            }
        } /* if (q_AN==0) */

        else {

            kg = natn[Gc_AN][0];
            Mk_AN = F_G2M[kg]; /* F_G2M should be used */
            wakg = WhatSpecies[kg];
            kan = Spe_Total_NO[wakg];
            kl1 = RMI1[Mc_AN][h_AN][0];
            kl2 = RMI1[Mc_AN][q_AN][0];

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    Re00x = 0.0;
                    Re00y = 0.0;
                    Re00z = 0.0;
                    Re11x = 0.0;
                    Re11y = 0.0;
                    Re11z = 0.0;
                    Re01x = 0.0;
                    Re01y = 0.0;
                    Re01z = 0.0;
                    Re10x = 0.0;
                    Re10y = 0.0;
                    Re10z = 0.0;

                    Im00x = 0.0;
                    Im00y = 0.0;
                    Im00z = 0.0;
                    Im11x = 0.0;
                    Im11y = 0.0;
                    Im11z = 0.0;
                    Im01x = 0.0;
                    Im01y = 0.0;
                    Im01z = 0.0;
                    Im10x = 0.0;
                    Im10y = 0.0;
                    Im10z = 0.0;

                    for (l1 = 0; l1 < kan; l1++) {
                        for (l2 = 0; l2 < kan; l2++) {

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].r;
                            Re00x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re00y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re00z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].r;
                            Re11x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re11y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re11z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].r;
                            Re01x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re01y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re01z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].r;
                            Re10x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Re10y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Re10z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][0][Mk_AN][l1][l2].i;
                            Im00x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im00y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im00z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][1][Mk_AN][l1][l2].i;
                            Im11x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im11y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im11z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[0][1][Mk_AN][l1][l2].i;
                            Im01x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im01y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im01z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];

                            ene = NC_v_eff[1][0][Mk_AN][l1][l2].i;
                            Im10x -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[1][Mj_AN][kl2][n][l2];
                            Im10y -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[2][Mj_AN][kl2][n][l2];
                            Im10z -= ene * OLP[0][Mi_AN][kl1][m][l1] * OLP[3][Mj_AN][kl2][n][l2];
                        }
                    }

                    Hx[0][0][m][n].r += Re00x;
                    Hx[0][0][m][n].i += Im00x;
                    Hy[0][0][m][n].r += Re00y;
                    Hy[0][0][m][n].i += Im00y;
                    Hz[0][0][m][n].r += Re00z;
                    Hz[0][0][m][n].i += Im00z;

                    Hx[1][1][m][n].r += Re11x;
                    Hx[1][1][m][n].i += Im11x;
                    Hy[1][1][m][n].r += Re11y;
                    Hy[1][1][m][n].i += Im11y;
                    Hz[1][1][m][n].r += Re11z;
                    Hz[1][1][m][n].i += Im11z;

                    Hx[0][1][m][n].r += Re01x;
                    Hx[0][1][m][n].i += Im01x;
                    Hy[0][1][m][n].r += Re01y;
                    Hy[0][1][m][n].i += Im01y;
                    Hz[0][1][m][n].r += Re01z;
                    Hz[0][1][m][n].i += Im01z;

                    Hx[1][0][m][n].r += Re10x;
                    Hx[1][0][m][n].i += Im10x;
                    Hy[1][0][m][n].r += Re10y;
                    Hy[1][0][m][n].i += Im10y;
                    Hz[1][0][m][n].r += Re10z;
                    Hz[1][0][m][n].i += Im10z;
                }
            }
        }
    }
}

#ifndef OPENMX_FORCE_PROJECTORS_H
#define OPENMX_FORCE_PROJECTORS_H
/* Shared host projector derivatives. Include after the native prototypes.
   Core-hole coefficients and angular traces belong to the C++ invocation. */

static void dHNL_block_term(const OpenMXProjectorForceView *projectors,size_t species,
    const OpenMXElectronicViews *electronic,int a,int ah,int b,int bh,int transpose,
    double sign,int reverse_energy,int value_first,OpenMXProjectorForceTerm *term)
{
    const OpenMXElectronicMatrixView *matrix=&electronic->fields[OPENMX_ELECTRONIC_PROJECTORS];
    const size_t ab=matrix->atom_offsets[matrix->component_offsets[0]+a]+ah;
    const size_t bb=matrix->atom_offsets[matrix->component_offsets[0]+b]+bh;
    memset(term,0,sizeof(*term));
    term->derivative_rows=matrix->shapes[2*ab];term->derivative_stride=matrix->shapes[2*ab+1];
    term->value_rows=matrix->shapes[2*bb];term->value_stride=matrix->shapes[2*bb+1];
    term->sign=sign;term->reverse_energy=reverse_energy;term->transpose=transpose;term->value_first=value_first;
    if(projectors->species_views[species].components)for(int c=0;c<=projectors->species_views[species].relativistic;++c) {
        if((size_t)c>=matrix->groups)OpenMX_Solver_Failure("missing nonlocal projector spin block");
        for(int axis=0;axis<3;++axis)term->derivative[c][axis]=openmx_electronic_row(matrix,c*matrix->channels+axis+1,a,ah,0);
        term->value[c]=openmx_electronic_row(matrix,c*matrix->channels,b,bh,0);
    }
}
static void dHNL_block(const OpenMXProjectorForceView *projectors,size_t species,
    const OpenMXProjectorForceTerm *terms,size_t count,size_t rows,size_t columns,size_t capacity,
    dcomplex ***hx,dcomplex ***hy,dcomplex ***hz,int accumulate)
{
    const OpenMXProjectorForceBlock block={rows,columns,capacity,capacity,count,terms,
        {hx[0][0],hy[0][0],hz[0][0]},accumulate};char error[256];
    if(!openmx_core_projector_force_block(projectors,species,&block,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

void dHNL(const OpenMXAtomicViews *atomic, const OpenMXProjectorForceView *projectors, int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double****** DS_NL1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz, const OpenMXElectronicViews *electronic)
{
  double *****HNL=electronic->HNL;
  double *****iHNL=electronic->iHNL;

    int i, j, k, m, n, l, kg, kan, so, deri_kind;
    int ig, ian, jg, jan, kl, kl1, kl2;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mi_AN2, Mj_AN, Mj_AN2;
    int Rni, Rnj, somax;
    double PF[2], sumx, sumy, sumz, ene, dmp, deri_dmp;
    double tmpx, tmpy, tmpz, tmp, r;
    double x0, y0, z0, x1, y1, z1, dx, dy, dz;
    double rcuti, rcutj, rcut;
    dcomplex sumx0, sumy0, sumz0;
    dcomplex sumx1, sumy1, sumz1;
    dcomplex sumx2, sumy2, sumz2;

    /****************************************************
     start calc.
    ****************************************************/

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Rni = ncn[Gc_AN][h_AN];
    Mi_AN = F_G2M[ig];
    ian = atomic->species.contracted_orbitals[WhatSpecies[ig]];
    rcuti = atomic->species.cutoff[WhatSpecies[ig]];

    jg = natn[Gc_AN][q_AN];
    Rnj = ncn[Gc_AN][q_AN];
    Mj_AN = F_G2M[jg];
    jan = atomic->species.contracted_orbitals[WhatSpecies[jg]];
    rcutj = atomic->species.cutoff[WhatSpecies[jg]];

    rcut = rcuti + rcutj;
    kl = RMI1[Mc_AN][h_AN][q_AN];
    dmp = dampingF(rcut, Dis[ig][kl]);

    /* every consumer of Hx/Hy/Hz reads only the ian x jan block of this
       pair (and the damping tail below scales the same block), so clearing
       the full List_YOUSO[7]^2 matrices would just stream ~100 KB per call */
    for (so = 0; so < 3; so++) {
        for (i = 0; i < ian; i++) {
            for (j = 0; j < jan; j++) {
                Hx[so][i][j] = Complex(0.0, 0.0);
                Hy[so][i][j] = Complex(0.0, 0.0);
                Hz[so][i][j] = Complex(0.0, 0.0);
            }
        }
    }

    if (h_AN == 0) {

        /****************************************************
                              dH*ep*H
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (0 <= kl && projectors->species_views[wakg].relativistic == 0 && where_flag == 0) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        if (Mj_AN <= Matomnum) Mj_AN2=Mj_AN; else Mj_AN2=Matomnum+1;
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Mc_AN,k,Mj_AN2,kl,0,1.0,0,0,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

                            l = 0;
                            for (l1 = 1; l1 <= (int)projectors->species_views[wakg].count; l1++) {

                                ene = projectors->species_views[wakg].channels[l1-1].energy[0];
                                l2 = projectors->species_views[wakg].channels[l1-1].twice_angular;

                                if (Mj_AN <= Matomnum)
                                    Mj_AN2 = Mj_AN;
                                else
                                    Mj_AN2 = Matomnum + 1;

                                for (l3 = 0; l3 <= l2; l3++) {
                                    sumx += ene * DS_NL1[0][1][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                    sumy += ene * DS_NL1[0][2][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                    sumz += ene * DS_NL1[0][3][Mc_AN][k][m][l] * DS_NL1[0][0][Mj_AN2][kl][n][l];
                                    l++;
                                }
                            }

                            Hx[0][m][n].r += sumx;
                            Hy[0][m][n].r += sumy;
                            Hz[0][m][n].r += sumz;

                            Hx[1][m][n].r += sumx;
                            Hy[1][m][n].r += sumy;
                            Hz[1][m][n].r += sumz;

                        } /* n */
                    }
                } /* m */

            } /* if */

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (0 <= kl && projectors->species_views[wakg].relativistic == 1 && where_flag == 0) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        if (Mj_AN <= Matomnum)
                            Mj_AN2 = Mj_AN;
                        else
                            Mj_AN2 = Matomnum + 1;
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Mc_AN,k,Mj_AN2,kl,0,1.0,0,0,&terms[0]);
                        if(q_AN==0) dHNL_block_term(projectors,wakg,electronic,Mj_AN2,kl,Mc_AN,k,1,-1.0,0,0,&terms[1]);
                        dHNL_block(projectors,wakg,terms,q_AN==0?2:1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx0 = Complex(0.0, 0.0);
                            sumy0 = Complex(0.0, 0.0);
                            sumz0 = Complex(0.0, 0.0);

                            sumx1 = Complex(0.0, 0.0);
                            sumy1 = Complex(0.0, 0.0);
                            sumz1 = Complex(0.0, 0.0);

                            sumx2 = Complex(0.0, 0.0);
                            sumy2 = Complex(0.0, 0.0);
                            sumz2 = Complex(0.0, 0.0);

                            if (Mj_AN <= Matomnum)
                                Mj_AN2 = Mj_AN;
                            else
                                Mj_AN2 = Matomnum + 1;

                            dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    1.0,
                                    0,
                                    DS_NL1[0][1][Mc_AN][k][m], DS_NL1[0][2][Mc_AN][k][m], DS_NL1[0][3][Mc_AN][k][m],
                                    DS_NL1[1][1][Mc_AN][k][m], DS_NL1[1][2][Mc_AN][k][m], DS_NL1[1][3][Mc_AN][k][m],
                                    DS_NL1[0][0][Mj_AN2][kl][n], DS_NL1[1][0][Mj_AN2][kl][n]);

                            if (q_AN == 0) {

                                dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                        &sumx1.r, &sumy1.r, &sumz1.r,
                                        &sumx2.r, &sumy2.r, &sumz2.r,
                                        &sumx0.i, &sumy0.i, &sumz0.i,
                                        &sumx1.i, &sumy1.i, &sumz1.i,
                                        &sumx2.i, &sumy2.i, &sumz2.i,
                                        -1.0,
                                        0,
                                        DS_NL1[0][1][Mj_AN2][kl][n], DS_NL1[0][2][Mj_AN2][kl][n], DS_NL1[0][3][Mj_AN2][kl][n],
                                        DS_NL1[1][1][Mj_AN2][kl][n], DS_NL1[1][2][Mj_AN2][kl][n], DS_NL1[1][3][Mj_AN2][kl][n],
                                        DS_NL1[0][0][Mc_AN][k][m], DS_NL1[1][0][Mc_AN][k][m]);
                            }

                            Hx[0][m][n].r += sumx0.r; /* up-up */
                            Hy[0][m][n].r += sumy0.r; /* up-up */
                            Hz[0][m][n].r += sumz0.r; /* up-up */

                            Hx[1][m][n].r += sumx1.r; /* dn-dn */
                            Hy[1][m][n].r += sumy1.r; /* dn-dn */
                            Hz[1][m][n].r += sumz1.r; /* dn-dn */

                            Hx[2][m][n].r += sumx2.r; /* up-dn */
                            Hy[2][m][n].r += sumy2.r; /* up-dn */
                            Hz[2][m][n].r += sumz2.r; /* up-dn */

                            Hx[0][m][n].i += sumx0.i; /* up-up */
                            Hy[0][m][n].i += sumy0.i; /* up-up */
                            Hz[0][m][n].i += sumz0.i; /* up-up */

                            Hx[1][m][n].i += sumx1.i; /* dn-dn */
                            Hy[1][m][n].i += sumy1.i; /* dn-dn */
                            Hz[1][m][n].i += sumz1.i; /* dn-dn */

                            Hx[2][m][n].i += sumx2.i; /* up-dn */
                            Hy[2][m][n].i += sumy2.i; /* up-dn */
                            Hz[2][m][n].i += sumz2.i; /* up-dn */
                        }
                    }
                }
            }

        } /* k */

        /****************************************************
                               H*ep*dH
        ****************************************************/

        /* h_AN==0 && q_AN==0 */

        if (q_AN == 0 && projectors->species_views[wakg].relativistic == 0) {

            for (m = 0; m < ian; m++) {
                for (n = m; n < jan; n++) {

                    tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
                    Hx[0][m][n].r = tmpx;
                    Hx[0][n][m].r = tmpx;
                    Hx[1][m][n].r = tmpx;
                    Hx[1][n][m].r = tmpx;

                    tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
                    Hy[0][m][n].r = tmpy;
                    Hy[0][n][m].r = tmpy;
                    Hy[1][m][n].r = tmpy;
                    Hy[1][n][m].r = tmpy;

                    tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
                    Hz[0][m][n].r = tmpz;
                    Hz[0][n][m].r = tmpz;
                    Hz[1][m][n].r = tmpz;
                    Hz[1][n][m].r = tmpz;
                }
            }
        }

        else if (where_flag == 1) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (projectors->species_views[wakg].relativistic == 0) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        if (Mj_AN <= Matomnum) { Mj_AN2=Mj_AN; kl2=RMI1[Mc_AN][q_AN][0]; }
                        else { Mj_AN2=Matomnum+1; kl2=RMI1[Mc_AN][0][q_AN]; }
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Mj_AN2,kl2,Mc_AN,0,1,-1.0,0,1,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

                            if (Mj_AN <= Matomnum) {
                                Mj_AN2 = Mj_AN;
                                kl2 = RMI1[Mc_AN][q_AN][0];
                            } else {
                                Mj_AN2 = Matomnum + 1;
                                kl2 = RMI1[Mc_AN][0][q_AN];
                            }

                            l = 0;
                            for (l1 = 1; l1 <= (int)projectors->species_views[wakg].count; l1++) {

                                ene = projectors->species_views[wakg].channels[l1-1].energy[0];
                                l2 = projectors->species_views[wakg].channels[l1-1].twice_angular;

                                for (l3 = 0; l3 <= l2; l3++) {

                                    sumx -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][1][Mj_AN2][kl2][n][l];
                                    sumy -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][2][Mj_AN2][kl2][n][l];
                                    sumz -= ene * DS_NL1[0][0][Mc_AN][0][m][l] * DS_NL1[0][3][Mj_AN2][kl2][n][l];
                                    l++;
                                }
                            }

                            Hx[0][m][n].r += sumx;
                            Hy[0][m][n].r += sumy;
                            Hz[0][m][n].r += sumz;

                            Hx[1][m][n].r += sumx;
                            Hy[1][m][n].r += sumy;
                            Hz[1][m][n].r += sumz;
                        }
                    }
                }
            }

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (projectors->species_views[wakg].relativistic == 1) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        if (Mj_AN <= Matomnum) {
                            Mj_AN2 = Mj_AN;
                            kl2 = RMI1[Mc_AN][q_AN][0];
                        } else {
                            Mj_AN2 = Matomnum + 1;
                            kl2 = RMI1[Mc_AN][0][q_AN];
                        }
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Mj_AN2,kl2,Mc_AN,0,1,-1.0,1,0,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx0 = Complex(0.0, 0.0);
                            sumy0 = Complex(0.0, 0.0);
                            sumz0 = Complex(0.0, 0.0);

                            sumx1 = Complex(0.0, 0.0);
                            sumy1 = Complex(0.0, 0.0);
                            sumz1 = Complex(0.0, 0.0);

                            sumx2 = Complex(0.0, 0.0);
                            sumy2 = Complex(0.0, 0.0);
                            sumz2 = Complex(0.0, 0.0);

                            if (Mj_AN <= Matomnum) {
                                Mj_AN2 = Mj_AN;
                                kl2 = RMI1[Mc_AN][q_AN][0];
                            } else {
                                Mj_AN2 = Matomnum + 1;
                                kl2 = RMI1[Mc_AN][0][q_AN];
                            }

                            dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    -1.0,
                                    1,
                                    DS_NL1[0][1][Mj_AN2][kl2][n], DS_NL1[0][2][Mj_AN2][kl2][n], DS_NL1[0][3][Mj_AN2][kl2][n],
                                    DS_NL1[1][1][Mj_AN2][kl2][n], DS_NL1[1][2][Mj_AN2][kl2][n], DS_NL1[1][3][Mj_AN2][kl2][n],
                                    DS_NL1[0][0][Mc_AN][0][m], DS_NL1[1][0][Mc_AN][0][m]);

                            Hx[0][m][n].r += sumx0.r; /* up-up */
                            Hy[0][m][n].r += sumy0.r; /* up-up */
                            Hz[0][m][n].r += sumz0.r; /* up-up */

                            Hx[1][m][n].r += sumx1.r; /* dn-dn */
                            Hy[1][m][n].r += sumy1.r; /* dn-dn */
                            Hz[1][m][n].r += sumz1.r; /* dn-dn */

                            Hx[2][m][n].r += sumx2.r; /* up-dn */
                            Hy[2][m][n].r += sumy2.r; /* up-dn */
                            Hz[2][m][n].r += sumz2.r; /* up-dn */

                            Hx[0][m][n].i += sumx0.i; /* up-up */
                            Hy[0][m][n].i += sumy0.i; /* up-up */
                            Hz[0][m][n].i += sumz0.i; /* up-up */

                            Hx[1][m][n].i += sumx1.i; /* dn-dn */
                            Hy[1][m][n].i += sumy1.i; /* dn-dn */
                            Hz[1][m][n].i += sumz1.i; /* dn-dn */

                            Hx[2][m][n].i += sumx2.i; /* up-dn */
                            Hy[2][m][n].i += sumy2.i; /* up-dn */
                            Hz[2][m][n].i += sumz2.i; /* up-dn */
                        }
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else if (where_flag == 0) {

        /****************************************************
           H*ep*dH

           if (h_AN!=0 && where_flag==0)
           This happens
           only if
           ( projectors->spin==3
             &&
             (projectors->spin_orbit==1 || (Hub_U_switch==1 && F_U_flag==1)
               || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
               || Zeeman_NCO_switch==1)
             &&
             q_AN==0
           )
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl = RMI1[Mc_AN][h_AN][k];

            if (Mi_AN <= Matomnum)
                Mi_AN2 = Mi_AN;
            else
                Mi_AN2 = Matomnum + 1;

            if (0 <= kl && projectors->species_views[wakg].relativistic == 1) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Mj_AN,k,Mi_AN2,kl,1,-1.0,0,0,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx0 = Complex(0.0, 0.0);
                            sumy0 = Complex(0.0, 0.0);
                            sumz0 = Complex(0.0, 0.0);

                            sumx1 = Complex(0.0, 0.0);
                            sumy1 = Complex(0.0, 0.0);
                            sumz1 = Complex(0.0, 0.0);

                            sumx2 = Complex(0.0, 0.0);
                            sumy2 = Complex(0.0, 0.0);
                            sumz2 = Complex(0.0, 0.0);

                            dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    -1.0,
                                    0,
                                    DS_NL1[0][1][Mj_AN][k][n], DS_NL1[0][2][Mj_AN][k][n], DS_NL1[0][3][Mj_AN][k][n],
                                    DS_NL1[1][1][Mj_AN][k][n], DS_NL1[1][2][Mj_AN][k][n], DS_NL1[1][3][Mj_AN][k][n],
                                    DS_NL1[0][0][Mi_AN2][kl][m], DS_NL1[1][0][Mi_AN2][kl][m]);

                            Hx[0][m][n].r += sumx0.r; /* up-up */
                            Hy[0][m][n].r += sumy0.r; /* up-up */
                            Hz[0][m][n].r += sumz0.r; /* up-up */

                            Hx[1][m][n].r += sumx1.r; /* dn-dn */
                            Hy[1][m][n].r += sumy1.r; /* dn-dn */
                            Hz[1][m][n].r += sumz1.r; /* dn-dn */

                            Hx[2][m][n].r += sumx2.r; /* up-dn */
                            Hy[2][m][n].r += sumy2.r; /* up-dn */
                            Hz[2][m][n].r += sumz2.r; /* up-dn */

                            Hx[0][m][n].i += sumx0.i; /* up-up */
                            Hy[0][m][n].i += sumy0.i; /* up-up */
                            Hz[0][m][n].i += sumz0.i; /* up-up */

                            Hx[1][m][n].i += sumx1.i; /* dn-dn */
                            Hy[1][m][n].i += sumy1.i; /* dn-dn */
                            Hz[1][m][n].i += sumz1.i; /* dn-dn */

                            Hx[2][m][n].i += sumx2.i; /* up-dn */
                            Hy[2][m][n].i += sumy2.i; /* up-dn */
                            Hz[2][m][n].i += sumz2.i; /* up-dn */
                        }
                    }
                }
            }
        }

    }

    /* if (h_AN!=0 && where_flag==1) */

    else {

        /****************************************************
                               dH*ep*H
        ****************************************************/

        kg = natn[Gc_AN][0];
        wakg = WhatSpecies[kg];
        kan = atomic->species.projector_components[wakg];
        kl1 = RMI1[Mc_AN][0][h_AN];
        kl2 = RMI1[Mc_AN][0][q_AN];

        /****************************************************
                       l-dependent non-local part
        ****************************************************/

        if (projectors->species_views[wakg].relativistic == 0) {

            if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                if(ian>0 && jan>0) {
                    OpenMXProjectorForceTerm terms[2];
                    dHNL_block_term(projectors,wakg,electronic,Matomnum + 1,kl1,Matomnum + 1,kl2,0,-1.0,0,0,&terms[0]);
                    dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,0);
                }
            }else {
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx = 0.0;
                        sumy = 0.0;
                        sumz = 0.0;

                        l = 0;
                        for (l1 = 1; l1 <= (int)projectors->species_views[wakg].count; l1++) {

                            ene = projectors->species_views[wakg].channels[l1-1].energy[0];
                                l2 = projectors->species_views[wakg].channels[l1-1].twice_angular;

                            for (l3 = 0; l3 <= l2; l3++) {
                                sumx -= ene * DS_NL1[0][1][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                                sumy -= ene * DS_NL1[0][2][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                                sumz -= ene * DS_NL1[0][3][Matomnum + 1][kl1][m][l] * DS_NL1[0][0][Matomnum + 1][kl2][n][l];
                                l++;
                            }
                        }

                        Hx[0][m][n].r = sumx;
                        Hy[0][m][n].r = sumy;
                        Hz[0][m][n].r = sumz;

                        Hx[1][m][n].r = sumx;
                        Hy[1][m][n].r = sumy;
                        Hz[1][m][n].r = sumz;

                        Hx[2][m][n].r = 0.0;
                        Hy[2][m][n].r = 0.0;
                        Hz[2][m][n].r = 0.0;

                        Hx[0][m][n].i = 0.0;
                        Hy[0][m][n].i = 0.0;
                        Hz[0][m][n].i = 0.0;

                        Hx[1][m][n].i = 0.0;
                        Hy[1][m][n].i = 0.0;
                        Hz[1][m][n].i = 0.0;

                        Hx[2][m][n].i = 0.0;
                        Hy[2][m][n].i = 0.0;
                        Hz[2][m][n].i = 0.0;
                    }
                }
            }
        }

        /****************************************************
                     j-dependent non-local part
        ****************************************************/

        else if (projectors->species_views[wakg].relativistic == 1) {

            if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                if(ian>0 && jan>0) {
                    OpenMXProjectorForceTerm terms[2];
                    dHNL_block_term(projectors,wakg,electronic,Matomnum + 1,kl1,Matomnum + 1,kl2,0,1.0,1,0,&terms[0]);
                    dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,0);
                }
            }else {
                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);

                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);

                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                &sumx1.r, &sumy1.r, &sumz1.r,
                                &sumx2.r, &sumy2.r, &sumz2.r,
                                &sumx0.i, &sumy0.i, &sumz0.i,
                                &sumx1.i, &sumy1.i, &sumz1.i,
                                &sumx2.i, &sumy2.i, &sumz2.i,
                                1.0,
                                1,
                                DS_NL1[0][1][Matomnum + 1][kl1][m], DS_NL1[0][2][Matomnum + 1][kl1][m], DS_NL1[0][3][Matomnum + 1][kl1][m],
                                DS_NL1[1][1][Matomnum + 1][kl1][m], DS_NL1[1][2][Matomnum + 1][kl1][m], DS_NL1[1][3][Matomnum + 1][kl1][m],
                                DS_NL1[0][0][Matomnum + 1][kl2][n], DS_NL1[1][0][Matomnum + 1][kl2][n]);

                        Hx[0][m][n].r = sumx0.r; /* up-up */
                        Hy[0][m][n].r = sumy0.r; /* up-up */
                        Hz[0][m][n].r = sumz0.r; /* up-up */

                        Hx[1][m][n].r = sumx1.r; /* dn-dn */
                        Hy[1][m][n].r = sumy1.r; /* dn-dn */
                        Hz[1][m][n].r = sumz1.r; /* dn-dn */

                        Hx[2][m][n].r = sumx2.r; /* up-dn */
                        Hy[2][m][n].r = sumy2.r; /* up-dn */
                        Hz[2][m][n].r = sumz2.r; /* up-dn */

                        Hx[0][m][n].i = sumx0.i; /* up-up */
                        Hy[0][m][n].i = sumy0.i; /* up-up */
                        Hz[0][m][n].i = sumz0.i; /* up-up */

                        Hx[1][m][n].i = sumx1.i; /* dn-dn */
                        Hy[1][m][n].i = sumy1.i; /* dn-dn */
                        Hz[1][m][n].i = sumz1.i; /* dn-dn */

                        Hx[2][m][n].i = sumx2.i; /* up-dn */
                        Hy[2][m][n].i = sumy2.i; /* up-dn */
                        Hz[2][m][n].i = sumz2.i; /* up-dn */
                    }
                }
            }
        }

        /****************************************************
                               H*ep*dH
        ****************************************************/

        if (q_AN != 0) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl1 = RMI1[Mc_AN][0][h_AN];
            kl2 = RMI1[Mc_AN][0][q_AN];

            /****************************************************
                           l-dependent non-local part
            ****************************************************/

            if (projectors->species_views[wakg].relativistic == 0) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Matomnum + 1,kl2,Matomnum + 1,kl1,1,-1.0,0,1,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx = 0.0;
                            sumy = 0.0;
                            sumz = 0.0;

                            l = 0;
                            for (l1 = 1; l1 <= (int)projectors->species_views[wakg].count; l1++) {

                                ene = projectors->species_views[wakg].channels[l1-1].energy[0];
                                l2 = projectors->species_views[wakg].channels[l1-1].twice_angular;

                                for (l3 = 0; l3 <= l2; l3++) {
                                    sumx -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][1][Matomnum + 1][kl2][n][l];
                                    sumy -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][2][Matomnum + 1][kl2][n][l];
                                    sumz -= ene * DS_NL1[0][0][Matomnum + 1][kl1][m][l] * DS_NL1[0][3][Matomnum + 1][kl2][n][l];
                                    l++;
                                }
                            }

                            Hx[0][m][n].r += sumx;
                            Hy[0][m][n].r += sumy;
                            Hz[0][m][n].r += sumz;

                            Hx[1][m][n].r += sumx;
                            Hy[1][m][n].r += sumy;
                            Hz[1][m][n].r += sumz;
                        }
                    }
                }
            }

            /****************************************************
                          j-dependent non-local part
            ****************************************************/

            else if (projectors->species_views[wakg].relativistic == 1) {

                if(electronic->fields[OPENMX_ELECTRONIC_PROJECTORS].values) {
                    if(ian>0 && jan>0) {
                        OpenMXProjectorForceTerm terms[2];
                        dHNL_block_term(projectors,wakg,electronic,Matomnum + 1,kl2,Matomnum + 1,kl1,1,-1.0,1,0,&terms[0]);
                        dHNL_block(projectors,wakg,terms,1,ian,jan,List_YOUSO[7],Hx,Hy,Hz,1);
                    }
                }else {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx0 = Complex(0.0, 0.0);
                            sumy0 = Complex(0.0, 0.0);
                            sumz0 = Complex(0.0, 0.0);

                            sumx1 = Complex(0.0, 0.0);
                            sumy1 = Complex(0.0, 0.0);
                            sumz1 = Complex(0.0, 0.0);

                            sumx2 = Complex(0.0, 0.0);
                            sumy2 = Complex(0.0, 0.0);
                            sumz2 = Complex(0.0, 0.0);

                            dHNL_SO_rows(projectors,wakg,&sumx0.r, &sumy0.r, &sumz0.r,
                                    &sumx1.r, &sumy1.r, &sumz1.r,
                                    &sumx2.r, &sumy2.r, &sumz2.r,
                                    &sumx0.i, &sumy0.i, &sumz0.i,
                                    &sumx1.i, &sumy1.i, &sumz1.i,
                                    &sumx2.i, &sumy2.i, &sumz2.i,
                                    -1.0,
                                    1,
                                    DS_NL1[0][1][Matomnum + 1][kl2][n], DS_NL1[0][2][Matomnum + 1][kl2][n], DS_NL1[0][3][Matomnum + 1][kl2][n],
                                    DS_NL1[1][1][Matomnum + 1][kl2][n], DS_NL1[1][2][Matomnum + 1][kl2][n], DS_NL1[1][3][Matomnum + 1][kl2][n],
                                    DS_NL1[0][0][Matomnum + 1][kl1][m], DS_NL1[1][0][Matomnum + 1][kl1][m]);

                            Hx[0][m][n].r += sumx0.r; /* up-up */
                            Hy[0][m][n].r += sumy0.r; /* up-up */
                            Hz[0][m][n].r += sumz0.r; /* up-up */

                            Hx[1][m][n].r += sumx1.r; /* dn-dn */
                            Hy[1][m][n].r += sumy1.r; /* dn-dn */
                            Hz[1][m][n].r += sumz1.r; /* dn-dn */

                            Hx[2][m][n].r += sumx2.r; /* up-dn */
                            Hy[2][m][n].r += sumy2.r; /* up-dn */
                            Hz[2][m][n].r += sumz2.r; /* up-dn */

                            Hx[0][m][n].i += sumx0.i; /* up-up */
                            Hy[0][m][n].i += sumy0.i; /* up-up */
                            Hz[0][m][n].i += sumz0.i; /* up-up */

                            Hx[1][m][n].i += sumx1.i; /* dn-dn */
                            Hy[1][m][n].i += sumy1.i; /* dn-dn */
                            Hz[1][m][n].i += sumz1.i; /* dn-dn */

                            Hx[2][m][n].i += sumx2.i; /* up-dn */
                            Hy[2][m][n].i += sumy2.i; /* up-dn */
                            Hz[2][m][n].i += sumz2.i; /* up-dn */
                        }
                    }
                }
            }
        }

    } /* else */

    /****************************************************
                 contribution by dampingF
    ****************************************************/

    /* Qij * dH/dx  */

    for (so = 0; so < 3; so++) {
        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Hx[so][m][n].r = dmp * Hx[so][m][n].r;
                Hy[so][m][n].r = dmp * Hy[so][m][n].r;
                Hz[so][m][n].r = dmp * Hz[so][m][n].r;

                Hx[so][m][n].i = dmp * Hx[so][m][n].i;
                Hy[so][m][n].i = dmp * Hy[so][m][n].i;
                Hz[so][m][n].i = dmp * Hz[so][m][n].i;
            }
        }
    }

    /* dQij/dx * H */

    if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {

        if (h_AN == 0)
            kl = q_AN;
        else if (q_AN == 0)
            kl = h_AN;

        if (projectors->spin == 0)
            somax = 0;
        else if (projectors->spin == 1)
            somax = 1;
        else if (projectors->spin == 3)
            somax = 2;

        r = Dis[Gc_AN][kl];

        if (rcut <= r) {
            deri_dmp = 0.0;
            tmp = 0.0;
        } else {
            deri_dmp = deri_dampingF(rcut, r);
            tmp = deri_dmp / dmp;
        }

        x0 = Gxyz[ig][1] + atv[Rni][1];
        y0 = Gxyz[ig][2] + atv[Rni][2];
        z0 = Gxyz[ig][3] + atv[Rni][3];

        x1 = Gxyz[jg][1] + atv[Rnj][1];
        y1 = Gxyz[jg][2] + atv[Rnj][2];
        z1 = Gxyz[jg][3] + atv[Rnj][3];

        /* for empty atoms or finite elemens basis */
        if (r < 1.0e-10)
            r = 1.0e-10;

        if (h_AN == 0 && q_AN != 0) {
            dx = tmp * (x0 - x1) / r;
            dy = tmp * (y0 - y1) / r;
            dz = tmp * (z0 - z1) / r;
        }

        else if (h_AN != 0 && q_AN == 0) {
            dx = tmp * (x1 - x0) / r;
            dy = tmp * (y1 - y0) / r;
            dz = tmp * (z1 - z0) / r;
        }

        if (projectors->spin == 0 || projectors->spin == 1) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }
        }

        else if (projectors->spin == 3) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HNL[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }

            if (projectors->spin_orbit == 1) {

                if (h_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dx;
                                Hy[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dy;
                                Hz[so][m][n].i += iHNL[so][Mc_AN][kl][m][n] * dz;
                            }
                        }
                    }
                }

                else if (q_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dx;
                                Hy[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dy;
                                Hz[so][m][n].i += iHNL[so][Mc_AN][kl][n][m] * dz;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void dHNL_SO_rows(const OpenMXProjectorForceView *projectors,size_t species,
    double* sumx0r,
    double* sumy0r,
    double* sumz0r,
    double* sumx1r,
    double* sumy1r,
    double* sumz1r,
    double* sumx2r,
    double* sumy2r,
    double* sumz2r,
    double* sumx0i,
    double* sumy0i,
    double* sumz0i,
    double* sumx1i,
    double* sumy1i,
    double* sumz1i,
    double* sumx2i,
    double* sumy2i,
    double* sumz2i,
    double fugou,int reverse_energy,
    const double* restrict A0x, const double* restrict A0y, const double* restrict A0z,
    const double* restrict A1x, const double* restrict A1y, const double* restrict A1z,
    const double* restrict B0, const double* restrict B1)
{
    dcomplex sum[9]={
        {*sumx0r,*sumx0i},{*sumy0r,*sumy0i},{*sumz0r,*sumz0i},
        {*sumx1r,*sumx1i},{*sumy1r,*sumy1i},{*sumz1r,*sumz1i},
        {*sumx2r,*sumx2i},{*sumy2r,*sumy2i},{*sumz2r,*sumz2i}
    };
    const OpenMXProjectorForceRows input={{{A0x,A0y,A0z},{A1x,A1y,A1z}},
        {B0,B1},projectors->species_views[species].components,fugou,reverse_energy};
    char error[256];
    if(!openmx_core_projector_force_trace(projectors,species,&input,sum,error,sizeof(error)))OpenMX_Solver_Failure(error);
    *sumx0r=sum[0].r;*sumx0i=sum[0].i;
    *sumy0r=sum[1].r;*sumy0i=sum[1].i;
    *sumz0r=sum[2].r;*sumz0i=sum[2].i;
    *sumx1r=sum[3].r;*sumx1i=sum[3].i;
    *sumy1r=sum[4].r;*sumy1i=sum[4].i;
    *sumz1r=sum[5].r;*sumz1i=sum[5].i;
    *sumx2r=sum[6].r;*sumx2i=sum[6].i;
    *sumy2r=sum[7].r;*sumy2i=sum[7].i;
    *sumz2r=sum[8].r;*sumz2i=sum[8].i;
}


void dHCH(const OpenMXAtomicViews *atomic, const OpenMXCoreHoleView *hole, int where_flag,
    int Mc_AN, int h_AN, int q_AN,
    double***** OLP1,
    dcomplex*** Hx, dcomplex*** Hy, dcomplex*** Hz, const OpenMXElectronicViews *electronic)
{
  double *****HCH=electronic->HCH;
  double *****iHCH=electronic->iHCH;

    int i, j, k, m, n, l, kg, kan, so, deri_kind, L;
    int ig, ian, jg, jan, kl, kl1, kl2, mul;
    int wakg, l1, l2, l3, Gc_AN, Mi_AN, Mi_AN2, Mj_AN, Mj_AN2;
    int Rni, Rnj, somax, apply_flag, target_spin;
    double PF[2], sumx[2], sumy[2], sumz[2], ene, dmp, deri_dmp;
    double tmpx, tmpy, tmpz, tmp, r;
    double x0, y0, z0, x1, y1, z1, dx, dy, dz;
    double rcuti, rcutj, rcut;
    double PFp, PFm, ene_p, ene_m;
    dcomplex sumx0, sumy0, sumz0;
    dcomplex sumx1, sumy1, sumz1;
    dcomplex sumx2, sumy2, sumz2;

    /****************************************************
     start calc.
    ****************************************************/

    const double *penalty[2]={hole->collinear[0],hole->collinear[1]};

    /* get information of relevant atoms */

    Gc_AN = M2G[Mc_AN];
    ig = natn[Gc_AN][h_AN];
    Rni = ncn[Gc_AN][h_AN];
    Mi_AN = F_G2M[ig];
    ian = atomic->species.contracted_orbitals[WhatSpecies[ig]];
    rcuti = atomic->species.cutoff[WhatSpecies[ig]];

    jg = natn[Gc_AN][q_AN];
    Rnj = ncn[Gc_AN][q_AN];
    Mj_AN = F_G2M[jg];
    jan = atomic->species.contracted_orbitals[WhatSpecies[jg]];
    rcutj = atomic->species.cutoff[WhatSpecies[jg]];

    rcut = rcuti + rcutj;
    kl = RMI1[Mc_AN][h_AN][q_AN];
    dmp = dampingF(rcut, Dis[ig][kl]);

    /* every consumer of Hx/Hy/Hz reads only the ian x jan block of this
       pair (and the damping tail below scales the same block), so clearing
       the full List_YOUSO[7]^2 matrices would just stream ~100 KB per call */
    for (so = 0; so < 3; so++) {
        for (i = 0; i < ian; i++) {
            for (j = 0; j < jan; j++) {
                Hx[so][i][j] = Complex(0.0, 0.0);
                Hy[so][i][j] = Complex(0.0, 0.0);
                Hz[so][i][j] = Complex(0.0, 0.0);
            }
        }
    }

    if (h_AN == 0) {

        /****************************************************
                              dH*ep*H
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = atomic->species.contracted_orbitals[wakg];
            kl = RMI1[Mc_AN][q_AN][k];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (0 <= kl && atomic->species.relativistic[wakg] == 0 && where_flag == 0) {

                if (Mj_AN <= Matomnum)
                    Mj_AN2 = Mj_AN;
                else
                    Mj_AN2 = Matomnum + 1;

                if (kg == hole->atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < atomic->species.contracted_orbitals[wakg]; i++) {
                                sumx[0] += penalty[0][i] * OLP1[1][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumy[0] += penalty[0][i] * OLP1[2][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumz[0] += penalty[0][i] * OLP1[3][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];

                                sumx[1] += penalty[1][i] * OLP1[1][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumy[1] += penalty[1][i] * OLP1[2][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                                sumz[1] += penalty[1][i] * OLP1[3][Mc_AN][k][m][i] * OLP1[0][Mj_AN2][kl][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];

                        } /* n */
                    } /* m */

                } /* if (kg==hole->atom) */

            } /* if */

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (0 <= kl && atomic->species.relativistic[wakg] == 1 && where_flag == 0) {

                if (Mj_AN <= Matomnum)
                    Mj_AN2 = Mj_AN;
                else
                    Mj_AN2 = Matomnum + 1;

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            1.0,
                            Mc_AN, k, m,
                            Mj_AN2, kl, n,
                            kg, OLP1, hole, 0);

                        if (q_AN == 0) {

                            dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                                &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                                &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                                -1.0,
                                Mj_AN2, kl, n,
                                Mc_AN, k, m,
                                kg, OLP1, hole, 0);
                        }

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }

        } /* k */

        /****************************************************
                               H*ep*dH
        ****************************************************/

        /* h_AN==0 && q_AN==0 */

        if (q_AN == 0 && atomic->species.relativistic[wakg] == 0) {

            for (m = 0; m < ian; m++) {
                for (n = m; n < jan; n++) {

                    tmpx = Hx[0][m][n].r + Hx[0][n][m].r;
                    Hx[0][m][n].r = tmpx;
                    Hx[0][n][m].r = tmpx;

                    tmpy = Hy[0][m][n].r + Hy[0][n][m].r;
                    Hy[0][m][n].r = tmpy;
                    Hy[0][n][m].r = tmpy;

                    tmpz = Hz[0][m][n].r + Hz[0][n][m].r;
                    Hz[0][m][n].r = tmpz;
                    Hz[0][n][m].r = tmpz;

                    tmpx = Hx[1][m][n].r + Hx[1][n][m].r;
                    Hx[1][m][n].r = tmpx;
                    Hx[1][n][m].r = tmpx;

                    tmpy = Hy[1][m][n].r + Hy[1][n][m].r;
                    Hy[1][m][n].r = tmpy;
                    Hy[1][n][m].r = tmpy;

                    tmpz = Hz[1][m][n].r + Hz[1][n][m].r;
                    Hz[1][m][n].r = tmpz;
                    Hz[1][n][m].r = tmpz;
                }
            }
        }

        else if (where_flag == 1) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl = RMI1[Mc_AN][q_AN][0];

            /****************************************************
                         l-dependent non-local part
            ****************************************************/

            if (atomic->species.relativistic[wakg] == 0) {

                if (Mj_AN <= Matomnum) {
                    Mj_AN2 = Mj_AN;
                    kl2 = RMI1[Mc_AN][q_AN][0];
                } else {
                    Mj_AN2 = Matomnum + 1;
                    kl2 = RMI1[Mc_AN][0][q_AN];
                }

                if (kg == hole->atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < atomic->species.contracted_orbitals[wakg]; i++) {
                                sumx[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[1][Mj_AN2][kl2][n][i];
                                sumy[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[2][Mj_AN2][kl2][n][i];
                                sumz[0] -= penalty[0][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[3][Mj_AN2][kl2][n][i];

                                sumx[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[1][Mj_AN2][kl2][n][i];
                                sumy[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[2][Mj_AN2][kl2][n][i];
                                sumz[1] -= penalty[1][i] * OLP1[0][Mc_AN][0][m][i] * OLP1[3][Mj_AN2][kl2][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];
                        }
                    }

                } /* if (kg==hole->atom) */
            }

            /****************************************************
                         j-dependent non-local part
            ****************************************************/

            else if (atomic->species.relativistic[wakg] == 1) {

                if (Mj_AN <= Matomnum) {
                    Mj_AN2 = Mj_AN;
                    kl2 = RMI1[Mc_AN][q_AN][0];
                } else {
                    Mj_AN2 = Matomnum + 1;
                    kl2 = RMI1[Mc_AN][0][q_AN];
                }

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        /* 1 */

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Mj_AN2, kl2, n,
                            Mc_AN, 0, m,
                            kg, OLP1, hole, 1);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* if (h_AN==0) */

    else if (where_flag == 0) {

        /****************************************************
           H*ep*dH

           if (h_AN!=0 && where_flag==0)
           This happens
           only if
           ( hole->spin==3
             &&
             (hole->spin_orbit==1 || (Hub_U_switch==1 && F_U_flag==1)
               || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1
               || Zeeman_NCO_switch==1)
             &&
             q_AN==0
           )
        ****************************************************/

        for (k = 0; k <= FNAN[Gc_AN]; k++) {

            kg = natn[Gc_AN][k];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl = RMI1[Mc_AN][h_AN][k];

            if (Mi_AN <= Matomnum)
                Mi_AN2 = Mi_AN;
            else
                Mi_AN2 = Matomnum + 1;

            if (0 <= kl && atomic->species.relativistic[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Mj_AN, k, n,
                            Mi_AN2, kl, m,
                            kg, OLP1, hole, 0);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    }

    /* if (h_AN!=0 && where_flag==1) */

    else {

        /****************************************************
                               dH*ep*H
        ****************************************************/

        kg = natn[Gc_AN][0];
        wakg = WhatSpecies[kg];
        kan = atomic->species.projector_components[wakg];
        kl1 = RMI1[Mc_AN][0][h_AN];
        kl2 = RMI1[Mc_AN][0][q_AN];

        /****************************************************
                       l-dependent non-local part
        ****************************************************/

        if (atomic->species.relativistic[wakg] == 0) {

            if (kg == hole->atom) {

                /* calculate the multiplication */

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx[0] = 0.0;
                        sumx[1] = 0.0;
                        sumy[0] = 0.0;
                        sumy[1] = 0.0;
                        sumz[0] = 0.0;
                        sumz[1] = 0.0;

                        for (i = 0; i < atomic->species.contracted_orbitals[wakg]; i++) {

                            sumx[0] -= penalty[0][i] * OLP1[1][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumy[0] -= penalty[0][i] * OLP1[2][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumz[0] -= penalty[0][i] * OLP1[3][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];

                            sumx[1] -= penalty[1][i] * OLP1[1][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumy[1] -= penalty[1][i] * OLP1[2][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                            sumz[1] -= penalty[1][i] * OLP1[3][Matomnum + 1][kl1][m][i] * OLP1[0][Matomnum + 1][kl2][n][i];
                        }

                        Hx[0][m][n].r = sumx[0];
                        Hy[0][m][n].r = sumy[0];
                        Hz[0][m][n].r = sumz[0];

                        Hx[1][m][n].r = sumx[1];
                        Hy[1][m][n].r = sumy[1];
                        Hz[1][m][n].r = sumz[1];

                        Hx[2][m][n].r = 0.0;
                        Hy[2][m][n].r = 0.0;
                        Hz[2][m][n].r = 0.0;

                        Hx[0][m][n].i = 0.0;
                        Hy[0][m][n].i = 0.0;
                        Hz[0][m][n].i = 0.0;

                        Hx[1][m][n].i = 0.0;
                        Hy[1][m][n].i = 0.0;
                        Hz[1][m][n].i = 0.0;

                        Hx[2][m][n].i = 0.0;
                        Hy[2][m][n].i = 0.0;
                        Hz[2][m][n].i = 0.0;
                    }
                }
            }
        }

        /****************************************************
                     j-dependent non-local part
        ****************************************************/

        else if (atomic->species.relativistic[wakg] == 1) {

            for (m = 0; m < ian; m++) {
                for (n = 0; n < jan; n++) {

                    sumx0 = Complex(0.0, 0.0);
                    sumy0 = Complex(0.0, 0.0);
                    sumz0 = Complex(0.0, 0.0);
                    sumx1 = Complex(0.0, 0.0);
                    sumy1 = Complex(0.0, 0.0);
                    sumz1 = Complex(0.0, 0.0);
                    sumx2 = Complex(0.0, 0.0);
                    sumy2 = Complex(0.0, 0.0);
                    sumz2 = Complex(0.0, 0.0);

                    /* 2 */

                    dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                        &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                        &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                        1.0,
                        Matomnum + 1, kl1, m,
                        Matomnum + 1, kl2, n,
                        kg, OLP1, hole, 1);

                    Hx[0][m][n].r = sumx0.r; /* up-up */
                    Hy[0][m][n].r = sumy0.r; /* up-up */
                    Hz[0][m][n].r = sumz0.r; /* up-up */

                    Hx[1][m][n].r = sumx1.r; /* dn-dn */
                    Hy[1][m][n].r = sumy1.r; /* dn-dn */
                    Hz[1][m][n].r = sumz1.r; /* dn-dn */

                    Hx[2][m][n].r = sumx2.r; /* up-dn */
                    Hy[2][m][n].r = sumy2.r; /* up-dn */
                    Hz[2][m][n].r = sumz2.r; /* up-dn */

                    Hx[0][m][n].i = sumx0.i; /* up-up */
                    Hy[0][m][n].i = sumy0.i; /* up-up */
                    Hz[0][m][n].i = sumz0.i; /* up-up */

                    Hx[1][m][n].i = sumx1.i; /* dn-dn */
                    Hy[1][m][n].i = sumy1.i; /* dn-dn */
                    Hz[1][m][n].i = sumz1.i; /* dn-dn */

                    Hx[2][m][n].i = sumx2.i; /* up-dn */
                    Hy[2][m][n].i = sumy2.i; /* up-dn */
                    Hz[2][m][n].i = sumz2.i; /* up-dn */
                }
            }
        }

        /****************************************************
                               H*ep*dH
        ****************************************************/

        if (q_AN != 0) {

            kg = natn[Gc_AN][0];
            wakg = WhatSpecies[kg];
            kan = atomic->species.projector_components[wakg];
            kl1 = RMI1[Mc_AN][0][h_AN];
            kl2 = RMI1[Mc_AN][0][q_AN];

            /****************************************************
                           l-dependent non-local part
            ****************************************************/

            if (atomic->species.relativistic[wakg] == 0) {

                if (kg == hole->atom) {

                    /* calculate the multiplication */

                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {

                            sumx[0] = 0.0;
                            sumx[1] = 0.0;
                            sumy[0] = 0.0;
                            sumy[1] = 0.0;
                            sumz[0] = 0.0;
                            sumz[1] = 0.0;

                            for (i = 0; i < atomic->species.contracted_orbitals[wakg]; i++) {

                                sumx[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[1][Matomnum + 1][kl2][n][i];
                                sumy[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[2][Matomnum + 1][kl2][n][i];
                                sumz[0] -= penalty[0][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[3][Matomnum + 1][kl2][n][i];

                                sumx[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[1][Matomnum + 1][kl2][n][i];
                                sumy[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[2][Matomnum + 1][kl2][n][i];
                                sumz[1] -= penalty[1][i] * OLP1[0][Matomnum + 1][kl1][m][i] * OLP1[3][Matomnum + 1][kl2][n][i];
                            }

                            Hx[0][m][n].r += sumx[0];
                            Hy[0][m][n].r += sumy[0];
                            Hz[0][m][n].r += sumz[0];

                            Hx[1][m][n].r += sumx[1];
                            Hy[1][m][n].r += sumy[1];
                            Hz[1][m][n].r += sumz[1];
                        }
                    }
                }
            }

            /****************************************************
                          j-dependent non-local part
            ****************************************************/

            else if (atomic->species.relativistic[wakg] == 1) {

                for (m = 0; m < ian; m++) {
                    for (n = 0; n < jan; n++) {

                        sumx0 = Complex(0.0, 0.0);
                        sumy0 = Complex(0.0, 0.0);
                        sumz0 = Complex(0.0, 0.0);
                        sumx1 = Complex(0.0, 0.0);
                        sumy1 = Complex(0.0, 0.0);
                        sumz1 = Complex(0.0, 0.0);
                        sumx2 = Complex(0.0, 0.0);
                        sumy2 = Complex(0.0, 0.0);
                        sumz2 = Complex(0.0, 0.0);

                        /* 4 */

                        dHCH_SO(&sumx0.r, &sumx0.i, &sumy0.r, &sumy0.i, &sumz0.r, &sumz0.i,
                            &sumx1.r, &sumx1.i, &sumy1.r, &sumy1.i, &sumz1.r, &sumz1.i,
                            &sumx2.r, &sumx2.i, &sumy2.r, &sumy2.i, &sumz2.r, &sumz2.i,
                            -1.0,
                            Matomnum + 1, kl2, n,
                            Matomnum + 1, kl1, m,
                            kg, OLP1, hole, 1);

                        Hx[0][m][n].r += sumx0.r; /* up-up */
                        Hy[0][m][n].r += sumy0.r; /* up-up */
                        Hz[0][m][n].r += sumz0.r; /* up-up */

                        Hx[1][m][n].r += sumx1.r; /* dn-dn */
                        Hy[1][m][n].r += sumy1.r; /* dn-dn */
                        Hz[1][m][n].r += sumz1.r; /* dn-dn */

                        Hx[2][m][n].r += sumx2.r; /* up-dn */
                        Hy[2][m][n].r += sumy2.r; /* up-dn */
                        Hz[2][m][n].r += sumz2.r; /* up-dn */

                        Hx[0][m][n].i += sumx0.i; /* up-up */
                        Hy[0][m][n].i += sumy0.i; /* up-up */
                        Hz[0][m][n].i += sumz0.i; /* up-up */

                        Hx[1][m][n].i += sumx1.i; /* dn-dn */
                        Hy[1][m][n].i += sumy1.i; /* dn-dn */
                        Hz[1][m][n].i += sumz1.i; /* dn-dn */

                        Hx[2][m][n].i += sumx2.i; /* up-dn */
                        Hy[2][m][n].i += sumy2.i; /* up-dn */
                        Hz[2][m][n].i += sumz2.i; /* up-dn */
                    }
                }
            }
        }

    } /* else */

    /****************************************************
                 contribution by dampingF
    ****************************************************/

    /* Qij * dH/dx  */

    for (so = 0; so < 3; so++) {
        for (m = 0; m < ian; m++) {
            for (n = 0; n < jan; n++) {

                Hx[so][m][n].r = dmp * Hx[so][m][n].r;
                Hy[so][m][n].r = dmp * Hy[so][m][n].r;
                Hz[so][m][n].r = dmp * Hz[so][m][n].r;

                Hx[so][m][n].i = dmp * Hx[so][m][n].i;
                Hy[so][m][n].i = dmp * Hy[so][m][n].i;
                Hz[so][m][n].i = dmp * Hz[so][m][n].i;
            }
        }
    }

    /* dQij/dx * H */

    if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {

        if (h_AN == 0)
            kl = q_AN;
        else if (q_AN == 0)
            kl = h_AN;

        if (hole->spin == 0)
            somax = 0;
        else if (hole->spin == 1)
            somax = 1;
        else if (hole->spin == 3)
            somax = 2;

        r = Dis[Gc_AN][kl];

        if (rcut <= r) {
            deri_dmp = 0.0;
            tmp = 0.0;
        } else {
            deri_dmp = deri_dampingF(rcut, r);
            tmp = deri_dmp / dmp;
        }

        x0 = Gxyz[ig][1] + atv[Rni][1];
        y0 = Gxyz[ig][2] + atv[Rni][2];
        z0 = Gxyz[ig][3] + atv[Rni][3];

        x1 = Gxyz[jg][1] + atv[Rnj][1];
        y1 = Gxyz[jg][2] + atv[Rnj][2];
        z1 = Gxyz[jg][3] + atv[Rnj][3];

        /* for empty atoms or finite elemens basis */
        if (r < 1.0e-10)
            r = 1.0e-10;

        if (h_AN == 0 && q_AN != 0) {
            dx = tmp * (x0 - x1) / r;
            dy = tmp * (y0 - y1) / r;
            dz = tmp * (z0 - z1) / r;
        }

        else if (h_AN != 0 && q_AN == 0) {
            dx = tmp * (x1 - x0) / r;
            dy = tmp * (y1 - y0) / r;
            dz = tmp * (z1 - z0) / r;
        }

        if (hole->spin == 0 || hole->spin == 1) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }
        }

        else if (hole->spin == 3) {

            if (h_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][m][n] * dz;
                        }
                    }
                }
            }

            else if (q_AN == 0) {
                for (so = 0; so <= somax; so++) {
                    for (m = 0; m < ian; m++) {
                        for (n = 0; n < jan; n++) {
                            Hx[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dx;
                            Hy[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dy;
                            Hz[so][m][n].r += HCH[so][Mc_AN][kl][n][m] * dz;
                        }
                    }
                }
            }

            if (hole->spin_orbit == 1) {

                if (h_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dx;
                                Hy[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dy;
                                Hz[so][m][n].i += iHCH[so][Mc_AN][kl][m][n] * dz;
                            }
                        }
                    }
                }

                else if (q_AN == 0) {
                    for (so = 0; so <= somax; so++) {
                        for (m = 0; m < ian; m++) {
                            for (n = 0; n < jan; n++) {
                                Hx[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dx;
                                Hy[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dy;
                                Hz[so][m][n].i += iHCH[so][Mc_AN][kl][n][m] * dz;
                            }
                        }
                    }
                }
            }
        }
    }

    /* freeing of array */

}

void dHCH_SO(double* sumx0r, double* sumx0i, double* sumy0r, double* sumy0i, double* sumz0r, double* sumz0i,
    double* sumx1r, double* sumx1i, double* sumy1r, double* sumy1i, double* sumz1r, double* sumz1i,
    double* sumx2r, double* sumx2i, double* sumy2r, double* sumy2i, double* sumz2r, double* sumz2i,
    double fugou,
    int Mc_AN, int k, int m,
    int Mj_AN, int kl, int n,
    int kg,
    double***** OLP1, const OpenMXCoreHoleView *hole, int reverse_penalty)
{
    if(kg!=hole->atom || !hole->active)return;
    dcomplex sum[9]={
        {*sumx0r,*sumx0i},{*sumy0r,*sumy0i},{*sumz0r,*sumz0i},
        {*sumx1r,*sumx1i},{*sumy1r,*sumy1i},{*sumz1r,*sumz1i},
        {*sumx2r,*sumx2i},{*sumy2r,*sumy2i},{*sumz2r,*sumz2i}
    };
    const OpenMXCoreHoleTrace input={{OLP1[1][Mc_AN][k][m],OLP1[2][Mc_AN][k][m],OLP1[3][Mc_AN][k][m]},
        OLP1[0][Mj_AN][kl][n],hole->orbitals,fugou,reverse_penalty};
    char error[256];
    if(!openmx_core_core_hole_trace(hole,&input,sum,error,sizeof(error)))OpenMX_Solver_Failure(error);
    *sumx0r=sum[0].r;*sumx0i=sum[0].i;
    *sumy0r=sum[1].r;*sumy0i=sum[1].i;
    *sumz0r=sum[2].r;*sumz0i=sum[2].i;
    *sumx1r=sum[3].r;*sumx1i=sum[3].i;
    *sumy1r=sum[4].r;*sumy1i=sum[4].i;
    *sumz1r=sum[5].r;*sumz1i=sum[5].i;
    *sumx2r=sum[6].r;*sumx2i=sum[6].i;
    *sumy2r=sum[7].r;*sumy2i=sum[7].i;
    *sumz2r=sum[8].r;*sumz2i=sum[8].i;
}

#endif

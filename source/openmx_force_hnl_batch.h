#ifndef OPENMX_FORCE_HNL_BATCH_H
#define OPENMX_FORCE_HNL_BATCH_H
#include "openmx_core_force_hnl_batch.h"
#include <stdint.h>
#include <limits.h>
/* Host preparation, recovery and damping are shared. Native dispatch retains
   each backend's original reduction and launch policy. The core callback scope
   owns all pair/row arrays; this frame only borrows views synchronously. */
typedef struct {
    int local_atoms,spin;
    const int *local_to_global,*global_to_local,*species,*orbitals,*neighbor_count;
    int *const *neighbors,*const *cells;
    int **const *pair_index;
    const double *cutoffs;
    double *const *distances,*const *positions,*const *translations;
} ForceHnlBatchInput;
typedef struct {
    int first,last,rank,solver_flat,profile,device;
    ForceHNLGpuContext *g;
    double *forces,*****CDM0;
    const OpenMXElectronicViews *electronic;
    char *arena;
    size_t energy_values;
    ForceHnlBatchInput input;
    double time[4];
} ForceHnlBatchWork;
static int Force_HNL_GpuCase1Dispatch(ForceHnlBatchWork*,const OpenMXForceHnlBatchView*);
static int Force_HNL_GpuCase2Dispatch(ForceHnlBatchWork*,const OpenMXForceHnlBatchView*);
static char *Force_HNL_BatchArena(ForceHnlBatchWork *work,size_t bytes)
{
    if(OpenMXGpuGetDevice(&work->device)!=OPENMX_GPU_SUCCESS)return NULL;
    work->arena=(char*)Force_HNL_GpuArena(work->g,bytes);
    return work->arena;
}
static void Force_HNL_BatchRelease(ForceHnlBatchWork *work)
{
    if(!work->arena)return;
    Force_HNL_GpuArenaFree(work->g,work->arena);work->arena=NULL;
    Force_gpu_pool_flush();
}
static void Force_HNL_BatchFinish(void *arguments)
{
    ForceHnlBatchWork *work=(ForceHnlBatchWork*)arguments;
    if(work->arena) {
        /* An interrupted dispatch must drain the device before C++ destroys
           the borrowed host arrays. Normal synchronous downloads release here
           before the damping tail, so add no fence to the successful path. */
        (void)Force_HNL_GpuDeviceSync(work->arena,work->device);
        Force_HNL_BatchRelease(work);
    }
}
static size_t Force_HNL_BatchCount(size_t count,size_t add)
{
    if(add>SIZE_MAX-count)Force4B_gpu_abort("Force_HNL density size overflow");
    return count+add;
}

static int Force_HNL_BatchCase1(const OpenMXForceHnlBatchView *view,void *arguments)
{
    ForceHnlBatchWork *work=(ForceHnlBatchWork*)arguments;
    ForceHNLGpuContext *g=work->g;
    double *****CDM0=work->CDM0,*****HNL=work->electronic->HNL;
    double *forces=work->forces;
    const int first=work->first,last=work->last,nitems=(int)view->items;
    const size_t cdm_count=view->density_values;
    const ForceHnlBatchInput *input=&work->input;
    const int Matomnum=input->local_atoms,SpinP_switch=input->spin;
    const int *M2G=input->local_to_global,*F_G2M=input->global_to_local;
    const int *WhatSpecies=input->species,*Spe_Total_CNO=input->orbitals,*FNAN=input->neighbor_count;
    int *const *natn=input->neighbors,*const *ncn=input->cells;
    int **const *RMI1=input->pair_index;
    const double *Spe_Atom_Cut1=input->cutoffs;
    double *const *Dis=input->distances,*const *Gxyz=input->positions,*const *atv=input->translations;
    Force4BGpuItem *items=view->item;
    int *item_atom=view->atom;
    double *cdm_pref=view->density,*item_f=view->force;
    int Mc_AN,q_AN,p=0;
    size_t cpos=0;
    const int kslots=(int)view->rows;
    int k,kpos=0,*item_q=view->q;
    size_t *krowA=view->a,*krowB=view->b;
    int *krow_halo=view->row_kind,*krow_ene=view->energy,*krow_nlp=view->projectors;
    /* build */
    p = 0;
    for (Mc_AN = first; Mc_AN < last; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            int Gq_AN, Mq_AN, jan, cdm_kl, i, j;
            const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;

            cdm_kl = RMI1[Mc_AN][0][q_AN];
            if (cdm_kl < 0) continue;

            Gq_AN = natn[Gc_AN][q_AN];
            Mq_AN = F_G2M[Gq_AN];
            jan = Spe_Total_CNO[WhatSpecies[Gq_AN]];

            items[p].cdm_off = cpos;
            items[p].k_off = kpos;
            items[p].ian = ian;
            items[p].jan = jan;
            item_atom[p] = Mc_AN;
            item_q[p] = q_AN;

            for (i = 0; i < ian; i++) {
                const double* c0 = CDM0[0][Mc_AN][cdm_kl][i];
                const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mc_AN][cdm_kl][i];

                for (j = 0; j < jan; j++) {
                    cdm_pref[cpos++] = pref * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                }
            }

            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                int qk_kl = RMI1[Mc_AN][q_AN][k];
                int kwan;

                if (qk_kl < 0) continue;
                kwan = WhatSpecies[natn[Gc_AN][k]];
                krowA[kpos] = g->mck_off[g->mck_base[Mc_AN] + k];
                krow_ene[kpos] = (int)g->ene_off[kwan];
                krow_nlp[kpos] = g->sp_nlp[kwan];
                if (Mq_AN <= Matomnum) {
                    krowB[kpos] = g->mck_off[g->mck_base[Mq_AN] + qk_kl];
                    krow_halo[kpos] = 0;
                } else {
                    krowB[kpos] = g->halo_off[g->halo_base[Mq_AN - Matomnum] + qk_kl];
                    krow_halo[kpos] = 1;
                }
                kpos++;
            }
            items[p].k_n = kpos - items[p].k_off;
            p++;
        }
    }

    if (p != nitems || kpos != kslots || cpos != cdm_count) {
        Force4B_gpu_abort("Force_HNL GPU: inconsistent first-case batch.");
    }

    int gpu_done=nitems==0;
    if(nitems && g->flat_dev)gpu_done=Force_HNL_GpuCase1Dispatch(work,view);
    Force_HNL_capacity(1,g,nitems,items,krowA,krowB,krow_halo,krow_ene,krow_nlp,NULL,cdm_pref,item_f,gpu_done);
    /* damping tail (dmp scaling plus the dQ/dx * HNL term) and the
       per-atom accumulation, exactly as the host dHNL applies them */

    for (p = 0; p < nitems; p++) {
        int mc = item_atom[p];
        int q_AN = item_q[p];
        int Gc_AN = M2G[mc];
        int Gq_AN = natn[Gc_AN][q_AN];
        int ian = items[p].ian;
        int jan = items[p].jan;
        int cdm_kl = RMI1[mc][0][q_AN];
        const double pref = (SpinP_switch == 0) ? 4.0 : 2.0;
        const double rcut = Spe_Atom_Cut1[WhatSpecies[Gc_AN]] + Spe_Atom_Cut1[WhatSpecies[Gq_AN]];
        const double dmp = dampingF(rcut, Dis[Gc_AN][cdm_kl]);
        double fx = item_f[3 * (size_t)p + 0] * dmp;
        double fy = item_f[3 * (size_t)p + 1] * dmp;
        double fz = item_f[3 * (size_t)p + 2] * dmp;
        double r = Dis[Gc_AN][q_AN];
        double tmp = 0.0;

        if (r < rcut) {
            tmp = deri_dampingF(rcut, r) / dmp;
        }
        if (r < 1.0e-10) {
            r = 1.0e-10;
        }

        if (tmp != 0.0) {
            const int Rni = ncn[Gc_AN][0];
            const int Rnj = ncn[Gc_AN][q_AN];
            const double dxv = tmp * ((Gxyz[Gc_AN][1] + atv[Rni][1]) - (Gxyz[Gq_AN][1] + atv[Rnj][1])) / r;
            const double dyv = tmp * ((Gxyz[Gc_AN][2] + atv[Rni][2]) - (Gxyz[Gq_AN][2] + atv[Rnj][2])) / r;
            const double dzv = tmp * ((Gxyz[Gc_AN][3] + atv[Rni][3]) - (Gxyz[Gq_AN][3] + atv[Rnj][3])) / r;
            const int somax = (SpinP_switch == 0) ? 0 : 1;
            double trace = 0.0;
            int so, i, j;

            for (so = 0; so <= somax; so++) {
                for (i = 0; i < ian; i++) {
                    const double* c = CDM0[so][mc][cdm_kl][i];
                    const double* hrow = HNL[so][mc][q_AN][i];

                    for (j = 0; j < jan; j++) {
                        trace += c[j] * hrow[j];
                    }
                }
            }
            trace *= pref;

            fx += trace * dxv;
            fy += trace * dyv;
            fz += trace * dzv;
        }

        if (forces) {
            forces[3*(size_t)Gc_AN]+=fx;forces[3*(size_t)Gc_AN+1]+=fy;forces[3*(size_t)Gc_AN+2]+=fz;
        } else {
        Gxyz[Gc_AN][41] += fx;
        Gxyz[Gc_AN][42] += fy;
        Gxyz[Gc_AN][43] += fz;
        }
    }

    return 1;
}

static int Force_HNL_BatchCase2(const OpenMXForceHnlBatchView *view,void *arguments)
{
    ForceHnlBatchWork *work=(ForceHnlBatchWork*)arguments;
    ForceHNLGpuContext *g=work->g;
    double *****CDM0=work->CDM0,*****HNL=work->electronic->HNL;
    double *forces=work->forces;
    const int first=work->first,last=work->last,nitems=(int)view->items;
    const size_t cdm_count=view->density_values;
    const ForceHnlBatchInput *input=&work->input;
    const int SpinP_switch=input->spin;
    const int *M2G=input->local_to_global,*F_G2M=input->global_to_local;
    const int *WhatSpecies=input->species,*Spe_Total_CNO=input->orbitals,*FNAN=input->neighbor_count;
    int *const *natn=input->neighbors,*const *ncn=input->cells;
    int **const *RMI1=input->pair_index;
    const double *Spe_Atom_Cut1=input->cutoffs;
    double *const *Dis=input->distances,*const *Gxyz=input->positions,*const *atv=input->translations;
    Force4BGpuItem *items=view->item;
    int *item_atom=view->atom;
    double *cdm_pref=view->density,*item_f=view->force;
    int Mc_AN,q_AN,p=0;
    size_t cpos=0;
    const int solver_flat=work->solver_flat;
    int h_AN,*item_h=view->h,*item_qn=view->q;
    int *item_h0=view->row_kind,*item_nlp=view->projectors,*item_ene=view->energy;
    size_t *item_a=view->a,*item_b=view->b;
    /* build */
    p = 0;
    for (Mc_AN = first; Mc_AN < last; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int Cwan = WhatSpecies[Gc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = solver_flat ? 0 : h_AN;
            int Gh_AN = natn[Gc_AN][h_AN];
            int Mh_AN = F_G2M[Gh_AN];
            int ian = Spe_Total_CNO[WhatSpecies[Gh_AN]];

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                int Gq_AN, jan, kl, kl1, kl2, i, j;
                double pref;

                if (h_AN == 0 && q_AN == 0) continue;
                kl = RMI1[Mc_AN][h_AN][q_AN];
                if (kl < 0) continue;

                Gq_AN = natn[Gc_AN][q_AN];
                jan = Spe_Total_CNO[WhatSpecies[Gq_AN]];

                if (solver_flat || q_AN == h_AN) {
                    pref = (SpinP_switch == 0) ? 2.0 : 1.0;
                } else {
                    pref = (SpinP_switch == 0) ? 4.0 : 2.0;
                }

                kl1 = RMI1[Mc_AN][0][h_AN];
                kl2 = RMI1[Mc_AN][0][q_AN];
                if (kl1 < 0 || kl2 < 0) {
                    Force4B_gpu_abort("Force_HNL GPU: negative staging index.");
                }

                items[p].cdm_off = cpos;
                items[p].k_off = 0;
                items[p].k_n = 0;
                items[p].ian = ian;
                items[p].jan = jan;
                item_atom[p] = Mc_AN;
                item_h[p] = h_AN;
                item_qn[p] = q_AN;
                item_h0[p] = (h_AN == 0) ? 1 : ((q_AN == 0) ? 2 : 0);
                item_nlp[p] = g->sp_nlp[Cwan];
                item_ene[p] = (int)g->ene_off[Cwan];
                if (h_AN == 0) {
                    /* value side: the centre's own k=0 row of the flat table */
                    item_a[p] = g->mck_off[g->mck_base[Mc_AN] + 0];
                    item_b[p] = g->c2_off[g->mck_base[Mc_AN] + kl2];
                } else {
                    item_a[p] = g->c2_off[g->mck_base[Mc_AN] + kl1];
                    item_b[p] = g->c2_off[g->mck_base[Mc_AN] + kl2];
                }

                for (i = 0; i < ian; i++) {
                    const double* c0 = CDM0[0][Mh_AN][kl][i];
                    const double* c1 = (SpinP_switch == 0) ? NULL : CDM0[1][Mh_AN][kl][i];

                    for (j = 0; j < jan; j++) {
                        cdm_pref[cpos++] = pref * ((c1 == NULL) ? c0[j] : (c0[j] + c1[j]));
                    }
                }
                p++;
            }
        }
    }

    if (p != nitems || cpos != cdm_count) {
        Force4B_gpu_abort("Force_HNL GPU: inconsistent second-case batch.");
    }

    if(work->profile)dtime(&work->time[1]);
    int gpu_done=nitems==0;
    if(nitems && g->flat_dev)gpu_done=Force_HNL_GpuCase2Dispatch(work,view);
    Force_HNL_capacity(2,g,nitems,items,item_a,item_b,NULL,item_ene,item_nlp,item_h0,cdm_pref,item_f,gpu_done);
    /* damping tail and per-centre accumulation */

    for (p = 0; p < nitems; p++) {
        int mc = item_atom[p];
        int h_AN = item_h[p];
        int q_AN = item_qn[p];
        int Gc_AN = M2G[mc];
        int Gh_AN = natn[Gc_AN][h_AN];
        int Gq_AN = natn[Gc_AN][q_AN];
        int ian = items[p].ian;
        int jan = items[p].jan;
        int kl = RMI1[mc][h_AN][q_AN];
        const double rcut = Spe_Atom_Cut1[WhatSpecies[Gh_AN]] + Spe_Atom_Cut1[WhatSpecies[Gq_AN]];
        const double dmp = dampingF(rcut, Dis[Gh_AN][kl]);
        double fx = item_f[3 * (size_t)p + 0] * dmp;
        double fy = item_f[3 * (size_t)p + 1] * dmp;
        double fz = item_f[3 * (size_t)p + 2] * dmp;

        if ((h_AN == 0 && q_AN != 0) || (h_AN != 0 && q_AN == 0)) {
            const int kld = (h_AN == 0) ? q_AN : h_AN;
            double r = Dis[Gc_AN][kld];
            double tmp = 0.0;

            if (r < rcut) {
                tmp = deri_dampingF(rcut, r) / dmp;
            }
            if (r < 1.0e-10) {
                r = 1.0e-10;
            }

            if (tmp != 0.0) {
                const int Rni = ncn[Gc_AN][h_AN];
                const int Rnj = ncn[Gc_AN][q_AN];
                const double sgn = (h_AN == 0) ? 1.0 : -1.0;
                const double dxv = sgn * tmp
                    * ((Gxyz[Gh_AN][1] + atv[Rni][1]) - (Gxyz[Gq_AN][1] + atv[Rnj][1])) / r;
                const double dyv = sgn * tmp
                    * ((Gxyz[Gh_AN][2] + atv[Rni][2]) - (Gxyz[Gq_AN][2] + atv[Rnj][2])) / r;
                const double dzv = sgn * tmp
                    * ((Gxyz[Gh_AN][3] + atv[Rni][3]) - (Gxyz[Gq_AN][3] + atv[Rnj][3])) / r;
                const int somax = (SpinP_switch == 0) ? 0 : 1;
                const int Mh_AN = F_G2M[Gh_AN];
                double pref;
                double trace = 0.0;
                int so, i, j;

                if (solver_flat || q_AN == h_AN) {
                    pref = (SpinP_switch == 0) ? 2.0 : 1.0;
                } else {
                    pref = (SpinP_switch == 0) ? 4.0 : 2.0;
                }

                for (so = 0; so <= somax; so++) {
                    for (i = 0; i < ian; i++) {
                        const double* c = CDM0[so][Mh_AN][kl][i];

                        if (h_AN == 0) {
                            const double* hrow = HNL[so][mc][kld][i];

                            for (j = 0; j < jan; j++) {
                                trace += c[j] * hrow[j];
                            }
                        } else {
                            for (j = 0; j < jan; j++) {
                                trace += c[j] * HNL[so][mc][kld][j][i];
                            }
                        }
                    }
                }
                trace *= pref;

                fx += trace * dxv;
                fy += trace * dyv;
                fz += trace * dzv;
            }
        }

        if (forces) {
            forces[3*(size_t)Gc_AN]+=fx;forces[3*(size_t)Gc_AN+1]+=fy;forces[3*(size_t)Gc_AN+2]+=fz;
        } else {
        Gxyz[Gc_AN][41] += fx;
        Gxyz[Gc_AN][42] += fy;
        Gxyz[Gc_AN][43] += fz;
        }
    }

    if(work->profile) {
        dtime(&work->time[3]);
        fprintf(stderr,"FORCEHNL2PROF id=%d items=%d build=%.3f device=%.3f post=%.3f\n",
            work->rank,nitems,work->time[1]-work->time[0],work->time[2]-work->time[1],work->time[3]-work->time[2]);
        fflush(stderr);
    }
    return 1;
}

static void Force_HNL_GpuCase1Run_Part(int first,int last,ForceHNLGpuContext *g,double *forces,int myid_p,double***** CDM0, const OpenMXElectronicViews *electronic)
{
    int Mc_AN,q_AN,nitems=0,kslots=0,k;
    size_t cdm_count=0;
    ForceHnlBatchWork work={0};
    /* count */
    for (Mc_AN = first; Mc_AN < last; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];

        for (q_AN = 1; q_AN <= FNAN[Gc_AN]; q_AN++) {
            if (RMI1[Mc_AN][0][q_AN] < 0) continue;
            if(nitems==INT_MAX)Force4B_gpu_abort("Force_HNL batch has too many pairs");
            nitems++;
            cdm_count=Force_HNL_BatchCount(cdm_count,(size_t)ian * (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]]);
            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                if (0 <= RMI1[Mc_AN][q_AN][k]) {
                    if(kslots==INT_MAX)Force4B_gpu_abort("Force_HNL batch has too many rows");
                    kslots++;
                }
            }
        }
    }

    work.first=first;work.last=last;work.g=g;work.forces=forces;work.rank=myid_p;
    work.CDM0=CDM0;work.electronic=electronic;
    work.input=(ForceHnlBatchInput){Matomnum,SpinP_switch,M2G,F_G2M,WhatSpecies,Spe_Total_CNO,FNAN,
        natn,ncn,RMI1,Spe_Atom_Cut1,Dis,Gxyz,atv};
    work.energy_values=g->ene_off[SpeciesNum];
    char error[256];
    if(!openmx_core_force_hnl_batch_run(1,(size_t)nitems,(size_t)kslots,cdm_count,
        Force_HNL_BatchCase1,Force_HNL_BatchFinish,&work,error,sizeof(error))) {
        fprintf(stderr,"Force_HNL batch: %s\n",error);
        Force4B_gpu_abort("nonlocal Force batch ownership failed");
    }
}

static void Force_HNL_GpuCase2Run_Part(int first,int last,ForceHNLGpuContext *g,double *forces,int myid_p,double***** CDM0, const OpenMXElectronicViews *electronic)
{
    const int solver_flat=(Solver==5 || Solver==8 || Solver==11);
    int Mc_AN,h_AN,q_AN,nitems=0;
    size_t cdm_count=0;
    ForceHnlBatchWork work={0};
    work.profile=Force_PhaseProfileEnabled();
    work.solver_flat=solver_flat;
    if(work.profile)dtime(&work.time[0]);
    /* count */
    for (Mc_AN = first; Mc_AN < last; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
            int start_q_AN = solver_flat ? 0 : h_AN;

            for (q_AN = start_q_AN; q_AN <= FNAN[Gc_AN]; q_AN++) {
                if (h_AN == 0 && q_AN == 0) continue;
                if (RMI1[Mc_AN][h_AN][q_AN] < 0) continue;
                if(nitems==INT_MAX)Force4B_gpu_abort("Force_HNL batch has too many pairs");
                nitems++;
                cdm_count=Force_HNL_BatchCount(cdm_count,(size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][h_AN]]]
                    * (size_t)Spe_Total_CNO[WhatSpecies[natn[Gc_AN][q_AN]]]);
            }
        }
    }

    work.first=first;work.last=last;work.g=g;work.forces=forces;work.rank=myid_p;
    work.CDM0=CDM0;work.electronic=electronic;
    work.input=(ForceHnlBatchInput){0,SpinP_switch,M2G,F_G2M,WhatSpecies,Spe_Total_CNO,FNAN,
        natn,ncn,RMI1,Spe_Atom_Cut1,Dis,Gxyz,atv};
    work.energy_values=g->ene_off[SpeciesNum];
    char error[256];
    if(!openmx_core_force_hnl_batch_run(2,(size_t)nitems,(size_t)nitems,cdm_count,
        Force_HNL_BatchCase2,Force_HNL_BatchFinish,&work,error,sizeof(error))) {
        fprintf(stderr,"Force_HNL batch: %s\n",error);
        Force4B_gpu_abort("nonlocal Force batch ownership failed");
    }
}

#endif

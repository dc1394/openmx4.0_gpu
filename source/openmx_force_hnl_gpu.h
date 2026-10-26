#ifndef OPENMX_FORCE_HNL_GPU_H
#define OPENMX_FORCE_HNL_GPU_H
#include "openmx_host_simd.h"
/* Invocation-owned nonlocal Force archives shared by CUDA/HIP. Native
   kernels and admission policies remain in their backend translation units. */
#include "openmx_native_resources.h"
#include "openmx_env.h"
#if OPENMX_GPU_HIP
#define OPENMX_HNL_WAIT() openmx_target_wait()
#define OPENMX_HNL_CLEAR() openmx_target_clear_freelists()
#define OPENMX_HNL_UPLOAD(d,h,n) openmx_target_upload(d,h,n)
#else
#define OPENMX_HNL_WAIT() acc_wait_all()
#define OPENMX_HNL_CLEAR() acc_clear_freelists()
#define OPENMX_HNL_UPLOAD(d,h,n) acc_memcpy_to_device(d,h,n)
#endif

struct ForceHNLGpuContext {
    int enabled;

    /* local DS_NL rows (so=0), direction-major: flat[dir*stride+block];
       the (Mc,k) block holds CNO(Gc) rows of nlp(species of natn) each */
    double* flat;
    double* flat_dev;
    size_t flat_stride;
    size_t* mck_off;
    int* mck_base;

    /* first-case halo archive: direction-0 rows of every received atom */
    double* halo;
    size_t halo_count;
    size_t* halo_off;
    int* halo_base;

    /* second-case per-centre archive: 4 directions of the staged slot;
       the (Mc,k) block holds CNO(natn[Gc][k]) rows of nlp(centre) each */
    double* c2;
    size_t c2_stride;
    size_t* c2_off;

    /* per-species expanded projector energies (one entry per l)        */
    double* ene;
    size_t* ene_off;
    int* sp_nlp;
    OpenMXNativeResources *resources;
};

static void *Force_HNL_GpuHost(ForceHNLGpuContext *g,size_t bytes)
{
    void *values=openmx_core_native_malloc(g->resources,1,bytes?bytes:1,0);
    if(!values)OpenMX_Solver_Failure("nonlocal Force host archive allocation failed");
    return values;
}
static int Force_HNL_GpuDeviceSync(void *pointer,int device)
{
    int previous,status=OpenMX_Resource_BindDevice(device,&previous);(void)pointer;
    if(status)return status;
    status=(int)OpenMXGpuDeviceSynchronize();
    OpenMX_Resource_RestoreDevice(device,previous);return status;
}
static void *Force_HNL_GpuArena(ForceHNLGpuContext *g,size_t bytes)
{
    void *pointer=Force_gpu_arena_try(bytes);
    if(!pointer)return NULL;
#if OPENMX_GPU_HIP
    if(!OpenMX_Resource_Adopt(g->resources,OPENMX_RESOURCE_DEVICE,pointer,
        OpenMX_Resource_DeviceFree,Force_HNL_GpuDeviceSync)) {
        (void)OpenMXGpuFree(pointer);return NULL;
    }
#else
    if(!OpenMX_Resource_Adopt(g->resources,OPENMX_RESOURCE_ACC,pointer,
        OpenMX_Resource_AccFreeCallback,Force_HNL_GpuDeviceSync)) {
        acc_free(pointer);return NULL;
    }
#endif
    return pointer;
}
static void Force_HNL_GpuArenaFree(ForceHNLGpuContext *g,void *pointer)
{
#if OPENMX_GPU_HIP
    (void)OpenMX_Resource_GpuFree(g->resources,pointer);
#else
    OpenMX_Resource_AccFree(g->resources,pointer);
#endif
}

static int Force_HNL_GpuBegin(const OpenMXAtomicViews *atomic, double****** DS_NL,
    OpenMXSolverInvocation *scratch,ForceHNLGpuContext **output)
{
    int Mc_AN, k, w, node_ranks = 1, node_rank = 0;
    size_t pos = 0, slots = 0, halo_slots = 0, ene_pos = 0;
    size_t free_bytes = 0, total_bytes = 0;
    MPI_Comm node_comm = MPI_COMM_NULL;

    int requested=scf_eigen_lib_flag==GPUSOLVER;
    MPI_Allreduce(MPI_IN_PLACE,&requested,1,MPI_INT,MPI_MIN,mpi_comm_level1);
    if (!requested) return 0;
    if (!openmx_env_flag_at(OPENMX_ENV_FORCEHNL_GPU, Force_gpu_stage_default())) return 0;
    if (F_NL_flag != 1) return 0;

    /* only the real scalar-relativistic trace branches are batched */
    if (!(SpinP_switch == 0 || SpinP_switch == 1 ||
          (SpinP_switch == 3 && SO_switch == 0 && Hub_U_switch == 0 &&
           Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0))) {
        return 0;
    }
    for (w = 0; w < SpeciesNum; w++) {
        if (atomic->species.relativistic[w] != 0) return 0;
    }

    OpenMXNativeResources *resources=OpenMX_Solver_Resources(scratch,"force-hnl-archives");
    void *state;char error[256];
    if(!openmx_core_native_state(resources,"archives",sizeof(ForceHNLGpuContext),&state,error,sizeof(error)))
        OpenMX_Solver_Failure(error);
    ForceHNLGpuContext *g=(ForceHNLGpuContext*)state;
    g->resources=resources;
    Force_HNL_GpuEnd(g);

    MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_ranks);
    MPI_Comm_rank(node_comm, &node_rank);
    if (node_ranks < 1) node_ranks = 1;
    if (gpu_rank_device_usable()) OPENMX_HNL_WAIT();
    if (gpu_rank_device_usable() && OpenMXGpuDeviceSynchronize() == OPENMX_GPU_SUCCESS) {
        OPENMX_HNL_CLEAR();
    }
    MPI_Barrier(node_comm);
    MPI_Comm_free(&node_comm);

    /* expanded projector energies per species */

    g->ene_off = (size_t*)Force_HNL_GpuHost(g,sizeof(size_t) * (size_t)(SpeciesNum + 1));
    g->sp_nlp = (int*)Force_HNL_GpuHost(g,sizeof(int) * (size_t)SpeciesNum);
    for (w = 0; w < SpeciesNum; w++) {
        int l1, nlp = 0;

        g->ene_off[w] = ene_pos;
        for (l1 = 1; l1 <= atomic->species.radial_projectors[w]; l1++) {
            nlp += 2 * openmx_atomic_projector_angular(atomic,w)[l1] + 1;
        }
        g->sp_nlp[w] = nlp;
        ene_pos += (size_t)nlp;
    }
    g->ene_off[SpeciesNum] = ene_pos;
    g->ene = (double*)Force_HNL_GpuHost(g,sizeof(double) * (ene_pos == 0 ? 1 : ene_pos));
    for (w = 0; w < SpeciesNum; w++) {
        int l1, l3, l = 0;

        for (l1 = 1; l1 <= atomic->species.radial_projectors[w]; l1++) {
            const double ene = openmx_radial_row3(&atomic->projector_energies,0,w)[l1 - 1];
            const int l2 = 2 * openmx_atomic_projector_angular(atomic,w)[l1];

            for (l3 = 0; l3 <= l2; l3++) {
                g->ene[g->ene_off[w] + (size_t)l] = ene;
                l++;
            }
        }
    }

    /* sizes of the flat local table, halo archive and staged archive */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int no0 = atomic->species.contracted_orbitals[WhatSpecies[Gc_AN]];

        slots += (size_t)(FNAN[Gc_AN] + 1);
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        }
    }
    g->flat_stride = pos;

    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        int no0 = atomic->species.contracted_orbitals[WhatSpecies[Gc_AN]];

        halo_slots += (size_t)(FNAN[Gc_AN] + 1);
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            g->halo_count += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        }
    }

    int resident=!openmx_gpu_capacity_test_recovery() && gpu_rank_device_usable() && OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)==OPENMX_GPU_SUCCESS;

    g->mck_base = (int*)Force_HNL_GpuHost(g,sizeof(int) * (size_t)(Matomnum + 2));
    g->mck_off = (size_t*)Force_HNL_GpuHost(g,sizeof(size_t) * (slots == 0 ? 1 : slots));
    g->c2_off = (size_t*)Force_HNL_GpuHost(g,sizeof(size_t) * (slots == 0 ? 1 : slots));

    pos = 0;
    slots = 0;
    {
        size_t c2pos = 0;

        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
            int Gc_AN = M2G[Mc_AN];
            int no0 = atomic->species.contracted_orbitals[WhatSpecies[Gc_AN]];
            int nlpc = g->sp_nlp[WhatSpecies[Gc_AN]];

            g->mck_base[Mc_AN] = (int)slots;
            for (k = 0; k <= FNAN[Gc_AN]; k++) {
                int nok = atomic->species.contracted_orbitals[WhatSpecies[natn[Gc_AN][k]]];

                g->mck_off[slots] = pos;
                g->c2_off[slots] = c2pos;
                pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
                c2pos += (size_t)nok * (size_t)nlpc;
                slots++;
            }
        }
        g->c2_stride = c2pos;
    }

    /* Host archives are required for remote and CPU recovery. Device
       residency is only an optimization, not admission to the phase. */
    {
        size_t need=(4U*g->flat_stride+g->halo_count+4U*g->c2_stride)*sizeof(double);
        unsigned long long room=openmx_host_mem_available();
        int fit=!room || need<=room/(2ULL*node_ranks),all_fit;
        MPI_Allreduce(&fit,&all_fit,1,MPI_INT,MPI_MIN,mpi_comm_level1);
        if(!all_fit){Force_HNL_GpuEnd(g);return 0;}
        size_t reserve=Force_gpu_node_reserve(node_ranks);
        /* Host SIMD policy (every rank reaches this point): need = the archives
           plus the 128 MiB transient margin of the residency test, budget = the
           per-rank share of the free memory above the node reserve. A host vote
           keeps the CPU dHNL traces (SIMD nonlocal blocks) on every rank. The
           capacity test forces the recovery: it does not vote. */
        {
            const size_t budget=resident && free_bytes>reserve ? (free_bytes-reserve)/(size_t)node_ranks : 0;
            if(OpenMX_HostSimdStageOnHost("Force_HNL",openmx_gpu_capacity_test_recovery() ? 0 : need+134217728,budget)){
                Force_HNL_GpuEnd(g);
                return 0;
            }
        }
        resident=resident && free_bytes>reserve && (free_bytes-reserve)/(size_t)node_ranks>need+134217728;
    }

    g->flat = (double*)Force_HNL_GpuHost(g,sizeof(double) * (g->flat_stride == 0 ? 4 : 4U * g->flat_stride));
    g->c2 = (double*)Force_HNL_GpuHost(g,sizeof(double) * (g->c2_stride == 0 ? 4 : 4U * g->c2_stride));
    g->halo_base = (int*)Force_HNL_GpuHost(g,sizeof(int) * (size_t)(MatomnumF + 2));
    g->halo_off = (size_t*)Force_HNL_GpuHost(g,sizeof(size_t) * (halo_slots == 0 ? 1 : halo_slots));
    g->halo = (double*)Force_HNL_GpuHost(g,sizeof(double) * (g->halo_count == 0 ? 1 : g->halo_count));

    halo_slots = 0;
    pos = 0;
    for (Mc_AN = Matomnum + 1; Mc_AN <= Matomnum + MatomnumF; Mc_AN++) {
        int Gc_AN = F_M2G[Mc_AN];
        int no0 = atomic->species.contracted_orbitals[WhatSpecies[Gc_AN]];

        g->halo_base[Mc_AN - Matomnum] = (int)halo_slots;
        for (k = 0; k <= FNAN[Gc_AN]; k++) {
            g->halo_off[halo_slots] = pos;
            pos += (size_t)no0 * (size_t)g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
            halo_slots++;
        }
    }

    /* flatten the local so=0 rows of all four directions */

#pragma omp parallel for schedule(dynamic)
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
        int Gc_AN = M2G[Mc_AN];
        int no0 = atomic->species.contracted_orbitals[WhatSpecies[Gc_AN]];
        int kk, k2;

        for (kk = 0; kk <= 3; kk++) {
            for (k2 = 0; k2 <= FNAN[Gc_AN]; k2++) {
                int nlp = g->sp_nlp[WhatSpecies[natn[Gc_AN][k2]]];
                size_t base = (size_t)kk * g->flat_stride
                    + g->mck_off[g->mck_base[Mc_AN] + k2];
                int i;

                for (i = 0; i < no0; i++) {
                    memcpy(g->flat + base + (size_t)i * (size_t)nlp,
                        DS_NL[0][kk][Mc_AN][k2][i], sizeof(double) * (size_t)nlp);
                }
            }
        }
    }

    if(resident){
        size_t flat_len=g->flat_stride?4U*g->flat_stride:4;
        g->flat_dev=(double*)Force_HNL_GpuArena(g,flat_len*sizeof(double));
        if(g->flat_dev)OPENMX_HNL_UPLOAD(g->flat_dev,g->flat,flat_len*sizeof(double));
    }

    g->enabled = 1;
    *output=g;return 1;
}

static void Force_HNL_GpuEnd(ForceHNLGpuContext *g)
{

    if (!g) return;

    if (g->flat_dev != NULL) {
        Force_HNL_GpuArenaFree(g,g->flat_dev);
        g->flat_dev = NULL;
    }
    if(g->enabled)OPENMX_HNL_CLEAR();

    openmx_core_native_free(g->resources,g->halo);
    openmx_core_native_free(g->resources,g->halo_off);
    openmx_core_native_free(g->resources,g->halo_base);
    openmx_core_native_free(g->resources,g->c2);
    openmx_core_native_free(g->resources,g->flat);
    openmx_core_native_free(g->resources,g->c2_off);
    openmx_core_native_free(g->resources,g->mck_off);
    openmx_core_native_free(g->resources,g->mck_base);
    openmx_core_native_free(g->resources,g->ene);
    openmx_core_native_free(g->resources,g->sp_nlp);
    openmx_core_native_free(g->resources,g->ene_off);
    OpenMXNativeResources *resources=g->resources;
    memset(g, 0, sizeof(*g));g->resources=resources;
}

static void Force_HNL_GpuArchiveHalo(ForceHNLGpuContext *g,double****** DS_NL, int Original_Mc_AN, int Gc_AN)
{
    int no0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    int halo_idx = Original_Mc_AN - Matomnum;
    int k;

    if (halo_idx < 1 || MatomnumF < halo_idx) {
        Force4B_gpu_abort("Force_HNL GPU: halo index out of range.");
    }

#pragma omp parallel for schedule(dynamic)
    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        int nlp = g->sp_nlp[WhatSpecies[natn[Gc_AN][k]]];
        size_t base = g->halo_off[g->halo_base[halo_idx] + k];
        int i;

        for (i = 0; i < no0; i++) {
            memcpy(g->halo + base + (size_t)i * (size_t)nlp,
                DS_NL[0][0][Matomnum + 1][k][i], sizeof(double) * (size_t)nlp);
        }
    }
}

static void Force_HNL_GpuArchiveCase2(ForceHNLGpuContext *g,double****** DS_NL, int Mc_AN, int Gc_AN)
{
    int nlpc = g->sp_nlp[WhatSpecies[Gc_AN]];
    int k;

#pragma omp parallel for schedule(dynamic)
    for (k = 0; k <= FNAN[Gc_AN]; k++) {
        int nok = Spe_Total_CNO[WhatSpecies[natn[Gc_AN][k]]];
        int kk, i;

        for (kk = 0; kk <= 3; kk++) {
            size_t base = (size_t)kk * g->c2_stride + g->c2_off[g->mck_base[Mc_AN] + k];

            for (i = 0; i < nok; i++) {
                memcpy(g->c2 + base + (size_t)i * (size_t)nlpc,
                    DS_NL[0][kk][Matomnum + 1][k][i], sizeof(double) * (size_t)nlpc);
            }
        }
    }
}

#undef OPENMX_HNL_WAIT
#undef OPENMX_HNL_CLEAR
#undef OPENMX_HNL_UPLOAD
#endif

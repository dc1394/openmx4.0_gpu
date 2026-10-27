/* Host-only MPI packing shared by all native backends. Numeric loops retain
   the collinear/noncollinear sparse order; no GPU runtime is required. */
#include "openmx_hamiltonian_pack.h"
#include "openmx_core_native_resources.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SET_HAMILTONIAN_PACK_ORDER_COL = 0, SET_HAMILTONIAN_PACK_ORDER_NONCOL = 1 };

struct OpenMXHamiltonianCache {
    OpenMXNativeResources *resources;
    int ready;
    int owns;
    int use_contracted;
    int order_mode;
    int size;
    int spin_count;
    int atom_count;
    int *order_GA;
    double *overlap;
    double *h[4];
    double *imnl[3];
};

static void packed_abort(MPI_Comm comm, const char *phase, const char *message, int rank)
{
    fprintf(stderr, "Set_Hamiltonian %s (rank %d): %s\n", phase, rank, message);
    MPI_Abort(comm, 1);
    abort();
}

static size_t packed_add(MPI_Comm comm, size_t a, size_t b, const char *name, int rank)
{
    if (a > SIZE_MAX - b) packed_abort(comm, name, "size overflow", rank);
    return a + b;
}

static size_t packed_mul(MPI_Comm comm, size_t a, size_t b, const char *name, int rank)
{
    if (a && b > SIZE_MAX / a) packed_abort(comm, name, "size overflow", rank);
    return a * b;
}

static void *packed_malloc(OpenMXHamiltonianCache *cache, MPI_Comm comm,
                           size_t bytes, const char *name, int rank)
{
    void *p = openmx_core_native_malloc(cache->resources, 1, bytes ? bytes : 1, 0);
    if (p == NULL) packed_abort(comm, name, "allocation failed", rank);
    return p;
}

static void Set_Hamiltonian_GpuSolver_Free_Cache(OpenMXHamiltonianCache *cache)
{
    OpenMXNativeResources *owner = cache->resources;
    int i;

    openmx_core_native_free(cache->resources, cache->order_GA);
    openmx_core_native_free(cache->resources, cache->overlap);
    for (i = 0; i < 4; i++) {
        openmx_core_native_free(cache->resources, cache->h[i]);
    }
    for (i = 0; i < 3; i++) {
        openmx_core_native_free(cache->resources, cache->imnl[i]);
    }

    memset(cache, 0, sizeof(*cache));
    cache->resources = owner;
}

void Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache(OpenMXHamiltonianCache *cache)
{
    Set_Hamiltonian_GpuSolver_Free_Cache(cache);
}

static int Set_Hamiltonian_GpuSolver_LocalPackedSize(const OpenMXHamiltonianPacking *input, MPI_Comm comm, int myid)
{
    size_t total = 0;

    for (int MA_AN = 1; MA_AN <= input->local_atom_count; MA_AN++) {
        int GA_AN = input->local_to_global[MA_AN];
        int wanA = input->species[GA_AN];
        int tnoA = input->orbitals[wanA];
        size_t neighbor_orbitals = 0;

        for (int LB_AN = 0; LB_AN <= input->neighbor_count[GA_AN]; LB_AN++) {
            int GB_AN = input->neighbors[GA_AN][LB_AN];
            int wanB = input->species[GB_AN];
            int tnoB = input->orbitals[wanB];

            neighbor_orbitals = packed_add(comm, neighbor_orbitals, (size_t)tnoB,
                                                            "GPU solver packed neighbor orbitals", myid);
        }

        total = packed_add(comm,
            total,
            packed_mul(comm, (size_t)tnoA, neighbor_orbitals,
                                        "GPU solver packed local matrix segment", myid),
            "GPU solver packed local matrix segment", myid);
    }

    if ((size_t)INT_MAX < total) {
        packed_abort(comm, "GPU solver packed cache", "local packed matrix segment exceeds INT_MAX", myid);
    }

    return (int)total;
}

static void Set_Hamiltonian_GpuSolver_PackLocalMatrix(const OpenMXHamiltonianPacking *input, MPI_Comm comm, double ****mat, double *local, int local_size, int order_mode,
                                                     int myid)
{
    int k = 0;

    if (mat == NULL) {
        packed_abort(comm, "GPU solver packed cache", "NULL sparse matrix", myid);
    }

    for (int MA_AN = 1; MA_AN <= input->local_atom_count; MA_AN++) {
        int GA_AN = input->local_to_global[MA_AN];
        int wanA = input->species[GA_AN];
        int tnoA = input->orbitals[wanA];

        if (order_mode == SET_HAMILTONIAN_PACK_ORDER_COL) {
            for (int LB_AN = 0; LB_AN <= input->neighbor_count[GA_AN]; LB_AN++) {
                int GB_AN = input->neighbors[GA_AN][LB_AN];
                int wanB = input->species[GB_AN];
                int tnoB = input->orbitals[wanB];

                for (int i = 0; i < tnoA; i++) {
                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
        else {
            for (int i = 0; i < tnoA; i++) {
                for (int LB_AN = 0; LB_AN <= input->neighbor_count[GA_AN]; LB_AN++) {
                    int GB_AN = input->neighbors[GA_AN][LB_AN];
                    int wanB = input->species[GB_AN];
                    int tnoB = input->orbitals[wanB];

                    for (int j = 0; j < tnoB; j++) {
                        local[k++] = mat[MA_AN][LB_AN][i][j];
                    }
                }
            }
        }
    }

    if (k != local_size) {
        packed_abort(comm, "GPU solver packed cache", "packed local matrix size mismatch", myid);
    }
}

static void Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(OpenMXHamiltonianCache *cache, const OpenMXHamiltonianPacking *input, MPI_Comm comm, double ****mat, double **target, int order_mode,
                                                             int local_size, int total_size, const int *counts,
                                                            const int *displs, const int *selected_roots,
                                                            MPI_Comm selected_comm, int first_selected_root,
                                                            int myid)
{
    double *local;
    double *recvbuf = NULL;

    *target = NULL;
    local = (double *)packed_malloc(cache, comm, sizeof(double) * (size_t)(0 < local_size ? local_size : 1),
                                             "GPU solver packed local matrix", myid);
    Set_Hamiltonian_GpuSolver_PackLocalMatrix(input, comm, mat, local, local_size, order_mode, myid);

    if (myid == first_selected_root) {
        recvbuf = (double *)packed_malloc(cache, comm, sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                   "GPU solver packed matrix cache", myid);
    }

    MPI_Gatherv(local, local_size, MPI_DOUBLE, recvbuf, (int *)counts, (int *)displs, MPI_DOUBLE, first_selected_root,
                comm);

    if (selected_roots[myid]) {
        if (myid == first_selected_root) {
            *target = recvbuf;
        }
        else {
            *target = (double *)packed_malloc(cache, comm, sizeof(double) * (size_t)(0 < total_size ? total_size : 1),
                                                       "GPU solver packed matrix cache", myid);
        }
        MPI_Bcast(*target, total_size, MPI_DOUBLE, 0, selected_comm);
    }

    openmx_core_native_free(cache->resources, local);
}

int Set_Hamiltonian_Build_GpuSolver_HS_Cache(OpenMXHamiltonianCache *cache,
    MPI_Comm comm, const OpenMXHamiltonianPacking *input, const OpenMXElectronicViews *electronic)
{
  double *****CntH=electronic->CntH;
  double *****CntOLP=electronic->CntOLP;
  double *****H=electronic->H;
  double *****OLP=electronic->OLP;
  double *****iCntHNL=electronic->iCntHNL;
  double *****iHNL=electronic->iHNL;

    int myid, numprocs;
    int local_selected;
    int selected_count = 0;
    int first_selected_root = -1;
    int local_size, total_size = 0;
    int local_matomnum;
    int *selected_roots = NULL;
    int *counts = NULL;
    int *displs = NULL;
    int *atom_counts = NULL;
    int *atom_displs = NULL;
    int dummy_atom = 0;
    int order_mode;
    int spin_count;
    int local_rebuild_static;
    int rebuild_static;
    MPI_Comm selected_comm = MPI_COMM_NULL;
    double ****overlap_src;
    double *****h_src;
    double *****imnl_src;

    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &myid);

    order_mode = (input->spin_mode == 3) ? SET_HAMILTONIAN_PACK_ORDER_NONCOL : SET_HAMILTONIAN_PACK_ORDER_COL;
    spin_count = (input->spin_mode == 3) ? 4 : (input->spin_mode + 1);
    local_selected = input->selected ? 1 : 0;
    local_size = Set_Hamiltonian_GpuSolver_LocalPackedSize(input, comm, myid);
    local_matomnum = input->local_atom_count;

    selected_roots = (int *)packed_malloc(cache, comm, sizeof(int) * (size_t)numprocs, "GPU solver selected roots", myid);
    counts = (int *)packed_malloc(cache, comm, sizeof(int) * (size_t)numprocs, "GPU solver packed counts", myid);
    displs = (int *)packed_malloc(cache, comm, sizeof(int) * (size_t)numprocs, "GPU solver packed displacements", myid);
    atom_counts = (int *)packed_malloc(cache, comm, sizeof(int) * (size_t)numprocs, "GPU solver atom counts", myid);
    atom_displs = (int *)packed_malloc(cache, comm, sizeof(int) * (size_t)numprocs, "GPU solver atom displacements", myid);

    MPI_Allgather(&local_selected, 1, MPI_INT, selected_roots, 1, MPI_INT, comm);
    MPI_Allgather(&local_size, 1, MPI_INT, counts, 1, MPI_INT, comm);
    MPI_Allgather(&local_matomnum, 1, MPI_INT, atom_counts, 1, MPI_INT, comm);

    for (int ID = 0; ID < numprocs; ID++) {
        if (selected_roots[ID]) {
            if (first_selected_root < 0) {
                first_selected_root = ID;
            }
            selected_count++;
        }
        if (counts[ID] < 0 || atom_counts[ID] < 0) {
            packed_abort(comm, "GPU solver packed cache", "negative gather count", myid);
        }
        if ((size_t)INT_MAX < (size_t)total_size + (size_t)counts[ID]) {
            packed_abort(comm, "GPU solver packed cache", "packed matrix size exceeds INT_MAX", myid);
        }
        displs[ID] = total_size;
        total_size += counts[ID];
    }

    {
        int atom_total = 0;
        for (int ID = 0; ID < numprocs; ID++) {
            if ((size_t)INT_MAX < (size_t)atom_total + (size_t)atom_counts[ID]) {
                packed_abort(comm, "GPU solver packed cache", "atom-order size exceeds INT_MAX", myid);
            }
            atom_displs[ID] = atom_total;
            atom_total += atom_counts[ID];
        }
        if (atom_total != input->atom_count) {
            packed_abort(comm, "GPU solver packed cache", "atom-order length mismatch", myid);
        }
    }

    if (selected_count == 0) {
        Set_Hamiltonian_GpuSolver_Free_Cache(cache);
        openmx_core_native_free(cache->resources, atom_displs);
        openmx_core_native_free(cache->resources, atom_counts);
        openmx_core_native_free(cache->resources, displs);
        openmx_core_native_free(cache->resources, counts);
        openmx_core_native_free(cache->resources, selected_roots);
        return 0;
    }

    /* All ranks must make the same decision before the optional order/overlap
       collectives.  A full refresh is mandatory at SCF 1 (new MD geometry),
       while later uncontracted SCFs can retain the invariant part. */
    rebuild_static = 1;
    if (input->reuse_static) {
      local_rebuild_static =
        input->refresh_static || !cache->ready || cache->use_contracted != (input->use_contracted ? 1 : 0) ||
        cache->order_mode != order_mode || cache->spin_count != spin_count || cache->atom_count != input->atom_count ||
        cache->size != total_size || cache->owns != local_selected ||
        (local_selected && (cache->order_GA == NULL || cache->overlap == NULL));
      MPI_Allreduce(&local_rebuild_static, &rebuild_static, 1, MPI_INT, MPI_MAX, comm);
    }

    if (rebuild_static) {
        Set_Hamiltonian_GpuSolver_Free_Cache(cache);
    }
    else {
        for (int spin = 0; spin < 4; spin++) {
            openmx_core_native_free(cache->resources, cache->h[spin]);
            cache->h[spin] = NULL;
        }
        for (int comp = 0; comp < 3; comp++) {
            openmx_core_native_free(cache->resources, cache->imnl[comp]);
            cache->imnl[comp] = NULL;
        }
        cache->ready = 0;
    }

    MPI_Comm_split(comm, local_selected ? 1 : MPI_UNDEFINED, myid, &selected_comm);

    cache->owns = local_selected;
    cache->use_contracted = input->use_contracted ? 1 : 0;
    cache->order_mode = order_mode;
    cache->size = total_size;
    cache->spin_count = spin_count;
    cache->atom_count = input->atom_count;

    if (rebuild_static) {
        if (myid == first_selected_root) {
            cache->order_GA = (int *)packed_malloc(cache, comm, sizeof(int) * ((size_t)input->atom_count + 2),
                                                            "GPU solver atom order cache", myid);
        }

        MPI_Gatherv((0 < input->local_atom_count) ? &input->local_to_global[1] : &dummy_atom, input->local_atom_count, MPI_INT,
                    (myid == first_selected_root) ? &cache->order_GA[1] : NULL,
                    atom_counts, atom_displs, MPI_INT, first_selected_root, comm);

        if (local_selected) {
            if (myid != first_selected_root) {
                cache->order_GA = (int *)packed_malloc(cache, comm, sizeof(int) * ((size_t)input->atom_count + 2),
                                                                "GPU solver atom order cache", myid);
            }
            cache->order_GA[0] = 0;
            cache->order_GA[input->atom_count + 1] = 0;
            MPI_Bcast(&cache->order_GA[1], input->atom_count, MPI_INT, 0, selected_comm);
        }
    }

    overlap_src = input->use_contracted ? CntOLP[0] : OLP[0];
    h_src = input->use_contracted ? CntH : H;
    imnl_src = input->use_contracted ? iCntHNL : iHNL;

    if (rebuild_static) {
        Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(cache, input, comm, overlap_src, &cache->overlap, order_mode, local_size,
                                                        total_size, counts, displs, selected_roots, selected_comm,
                                                        first_selected_root, myid);
    }

    for (int spin = 0; spin < spin_count; spin++) {
        Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(cache, input, comm, h_src[spin], &cache->h[spin], order_mode, local_size,
                                                        total_size, counts, displs, selected_roots, selected_comm,
                                                        first_selected_root, myid);
    }

    if (input->spin_mode == 3 && imnl_src != NULL) {
        for (int comp = 0; comp < 3; comp++) {
            Set_Hamiltonian_GpuSolver_GatherMatrixToSelected(cache, input, comm, imnl_src[comp], &cache->imnl[comp], order_mode,
                                                            local_size, total_size, counts, displs, selected_roots,
                                                            selected_comm, first_selected_root, myid);
        }
    }

    cache->ready = 1;

    if (selected_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&selected_comm);
    }

    openmx_core_native_free(cache->resources, atom_displs);
    openmx_core_native_free(cache->resources, atom_counts);
    openmx_core_native_free(cache->resources, displs);
    openmx_core_native_free(cache->resources, counts);
    openmx_core_native_free(cache->resources, selected_roots);
    return rebuild_static;
}

int Set_Hamiltonian_GpuSolver_Packed_CacheReady(const OpenMXHamiltonianCache *cache)
{
    return cache->ready;
}

int Set_Hamiltonian_GpuSolver_Packed_OwnsCache(const OpenMXHamiltonianCache *cache)
{
    return (cache->ready && cache->owns);
}

int Set_Hamiltonian_GpuSolver_Packed_OrderMode(const OpenMXHamiltonianCache *cache)
{
    return cache->order_mode;
}

int Set_Hamiltonian_GpuSolver_Packed_Size(const OpenMXHamiltonianCache *cache)
{
    return cache->size;
}

int *Set_Hamiltonian_GpuSolver_Packed_OrderGA(const OpenMXHamiltonianCache *cache)
{
    return cache->order_GA;
}

double *Set_Hamiltonian_GpuSolver_Packed_Overlap(const OpenMXHamiltonianCache *cache)
{
    return cache->overlap;
}

double *Set_Hamiltonian_GpuSolver_Packed_H(const OpenMXHamiltonianCache *cache, int spin)
{
    if (spin < 0 || 4 <= spin) {
        return NULL;
    }
    return cache->h[spin];
}

double *Set_Hamiltonian_GpuSolver_Packed_ImNL(const OpenMXHamiltonianCache *cache, int comp)
{
    if (comp < 0 || 3 <= comp) {
        return NULL;
    }
    return cache->imnl[comp];
}

static void packed_reset(OpenMXNativeResources *owner, void *metadata)
{
    (void)owner;
    Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache((OpenMXHamiltonianCache *)metadata);
}

OpenMXHamiltonianCache *Set_Hamiltonian_Cache_Bind_Resources(OpenMXCoreContext *core)
{
    OpenMXNativeResources *owner = NULL;
    void *metadata = NULL;
    char error[256];
    if (!openmx_core_native_resources_state(core, "Set_Hamiltonian_Packed",
        sizeof(OpenMXHamiltonianCache), packed_reset, &owner, &metadata, error, sizeof(error))) {
        fprintf(stderr, "OpenMX packed Hamiltonian cache: %s\n", error);
        abort();
    }
    OpenMXHamiltonianCache *cache = (OpenMXHamiltonianCache *)metadata;
    cache->resources = owner;
    return cache;
}

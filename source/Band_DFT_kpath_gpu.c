/* MPI and legacy matrix views for the post-SCF band path. The device backend
   consumes a compact, read-only copy of the real-space matrices. */
#include "openmx_common.h"
#include "openmx_band_kpath_gpu.h"
#include "openmx_band_kpath_points.h"
#include "openmx_gpu_device.h"
#include "openmx_host_memory.h"
#include "openmx_host_simd.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_env.h"
#if OPENMX_GPU_CUDA
#include "openmx_exact_exchange.h"
#endif

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int atoms, local_atoms;
    const int *local_to_global, *global_to_halo, *species, *paos, *lnos, *neighbor_counts;
    int *const *neighbors, *const *cells, *const *translations;
} KPathBasis;

static int all_ranks_ok(MPI_Comm comm, int local)
{
    int ready;
    MPI_Allreduce(&local, &ready, 1, MPI_INT, MPI_MIN, comm);
    return ready;
}

/* Elect once on each node, including jobs where ranks see different
   ordinals for the same card. Prefer the first usable rank on each GPU. */
static int device_owner(MPI_Comm comm)
{
    MPI_Comm node;
    int rank, size, device = -1;
    unsigned char identity[17] = {0};

    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &rank);
    MPI_Comm_size(node, &size);
    unsigned char (*peers)[17] = malloc((size_t)size * sizeof(*peers));
    int ready = peers != NULL, node_ready;
    MPI_Allreduce(&ready, &node_ready, 1, MPI_INT, MPI_MIN, node);
    if (node_ready) {
        if (gpu_rank_device_usable() && OpenMXGpuGetDevice(&device) == OPENMX_GPU_SUCCESS)
            identity[0] = openmx_gpu_physical_uuid(device, identity + 1);
        MPI_Allgather(identity, sizeof(identity), MPI_UNSIGNED_CHAR,
                      peers, sizeof(identity), MPI_UNSIGNED_CHAR, node);
        if (!identity[0]) device = -1;
        for (int i = 0; i < rank && device >= 0; ++i)
            if (memcmp(identity, peers[i], sizeof(identity)) == 0) device = -1;
    }
    free(peers);
    MPI_Comm_free(&node);
    return device;
}

static void interpolate(double **path, int point, int count, double xyz[3])
{
    for (int axis = 0; axis < 3; ++axis)
        xyz[axis] = openmx_band_kpath_coordinate(path, axis + 1, point, count);
}

static int compare_points(const void *left, const void *right)
{
    const OpenMXKPathPoint *a = left, *b = right;
    for (int axis = 0; axis < 3; ++axis) {
        if (a->xyz[axis] < b->xyz[axis]) return -1;
        if (a->xyz[axis] > b->xyz[axis]) return 1;
    }
    return 0;
}

/* Collapse exact repeats only. The map restores the original file order
   (including endpoints and repeated path segments). */
static int unique_points(int path_count, const int *counts, double ***paths,
                         OpenMXKPathPoint *points, int *output_to_unique, int count)
{
    int index = 0, unique = 0;
    for (int path = 1; path <= path_count; ++path) {
        for (int point = 0; point < counts[path]; ++point) {
            interpolate(paths[path], point, counts[path], points[index].xyz);
            for (int axis = 0; axis < 3; ++axis)
                if (!isfinite(points[index].xyz[axis])) return 0;
            points[index].output_index = index;
            ++index;
        }
    }
    qsort(points, count, sizeof(*points), compare_points);
    for (int i = 0; i < count; ++i) {
        const int output = points[i].output_index;
        if (unique == 0 || compare_points(&points[unique - 1], &points[i]) != 0)
            points[unique++] = points[i];
        output_to_unique[output] = unique - 1;
    }
    return unique;
}

static int values_per_orbital_pair(const OpenMXKPathMatrices *m)
{
    if (m->noncollinear) return m->imaginary_diagonals ? 7 : 5;
    return m->lno ? 2 * m->spin_channels : 1 + m->spin_channels;
}

static double lno_project(const double *left, double **block, const double *right,
                          int rows, int cols)
{
    double value = 0.0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            value += left[i] * block[i][j] * right[j];
    return value;
}

static void pack_values(const OpenMXKPathMatrices *m, const KPathBasis *basis, int atom, int neighbor,
                        int rows, int cols, double *****h, double *****ih, double ****s,
                        const OpenMXLocalBasisViews *local_basis, double *values)
{
    if (m->lno) {
        const int a = basis->local_to_global[atom], b = basis->neighbors[a][neighbor];
        const int local_b = basis->global_to_halo[b];
        const int paos_a = basis->paos[basis->species[a]], paos_b = basis->paos[basis->species[b]];
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                for (int spin = 0; spin < m->spin_channels; ++spin) {
                    const double *left = openmx_local_lno_row(local_basis, spin, atom) + i * paos_a;
                    const double *right = openmx_local_lno_row(local_basis, spin, local_b) + j * paos_b;
                    *values++ = lno_project(left, s[atom][neighbor], right, paos_a, paos_b);
                    *values++ = lno_project(left, h[spin][atom][neighbor], right, paos_a, paos_b);
                }
            }
        }
        return;
    }

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            *values++ = s[atom][neighbor][i][j];
            if (m->noncollinear) {
                *values++ = h[0][atom][neighbor][i][j];
                *values++ = h[1][atom][neighbor][i][j];
                *values++ = h[2][atom][neighbor][i][j];
                *values++ = h[3][atom][neighbor][i][j] +
                            (m->imaginary_diagonals ? ih[2][atom][neighbor][i][j] : 0.0);
                if (m->imaginary_diagonals) {
                    *values++ = ih[0][atom][neighbor][i][j];
                    *values++ = ih[1][atom][neighbor][i][j];
                }
            } else {
                for (int spin = 0; spin < m->spin_channels; ++spin)
                    *values++ = h[spin][atom][neighbor][i][j];
            }
        }
    }
}

static int orbital_offsets(const OpenMXKPathMatrices *m, const KPathBasis *basis, int *offset)
{
    offset[1] = 0;
    for (int atom = 1; atom <= basis->atoms; ++atom) {
        const int count = m->lno ? basis->lnos[atom] : basis->paos[basis->species[atom]];
        if (count < 0 || count > INT_MAX - offset[atom]) return 0;
        offset[atom + 1] = offset[atom] + count;
    }
    return offset[basis->atoms + 1];
}

/* The counting pass uses the same iterator as packing. The real-space
   payload is sent once per invocation, independent of the number of k-points. */
static int local_blocks(const OpenMXKPathMatrices *m, const KPathBasis *basis, const int *offset,
                        double *****h, double *****ih, double ****s,
                        const OpenMXLocalBasisViews *local_basis,
                        OpenMXKPathBlock *blocks, double *values, int counts[2])
{
    int count = 0, value_count = 0, width = values_per_orbital_pair(m);
    for (int atom = 1; atom <= basis->local_atoms; ++atom) {
        const int a = basis->local_to_global[atom], rows = offset[a + 1] - offset[a];
        for (int neighbor = 0; neighbor <= basis->neighbor_counts[a]; ++neighbor) {
            const int b = basis->neighbors[a][neighbor], cols = offset[b + 1] - offset[b];
            if (rows == 0 || cols == 0) continue;
            if (count >= INT_MAX / (int)sizeof(*blocks) ||
                (size_t)rows * cols > (size_t)(INT_MAX - value_count) / width) return 0;
            if (blocks) {
                const int *shift = basis->translations[basis->cells[a][neighbor]];
                blocks[count] = (OpenMXKPathBlock){
                    offset[a], offset[b], rows, cols, {shift[1], shift[2], shift[3]}, value_count
                };
                pack_values(m, basis, atom, neighbor, rows, cols, h, ih, s, local_basis, values + value_count);
            }
            ++count;
            value_count += rows * cols * width;
        }
    }
    counts[0] = count * (int)sizeof(*blocks);
    counts[1] = value_count;
    return 1;
}

#if OPENMX_GPU_CUDA
static int size_exchange(OpenMXKPathMatrices *m, const OpenMXExactExchangeOutputView *exchange,
                         int atoms, const int *offset)
{
    if (m->noncollinear || m->lno || exchange->atoms != atoms || exchange->orbitals != m->order ||
        exchange->spin_channels != m->spin_channels || exchange->blocks < 0 ||
        exchange->blocks > INT_MAX / (int)sizeof(OpenMXKPathBlock) - m->block_count) return 0;
    for (int i = 0; i <= atoms; ++i)
        if (exchange->first_orbital[i] != offset[i + 1]) return 0;
    const size_t elements = exchange->block_offset[exchange->blocks];
    if (elements > (size_t)(INT_MAX - m->value_count) / m->spin_channels) return 0;
    m->block_count += (int)exchange->blocks;
    m->value_count += (int)elements * m->spin_channels;
    return 1;
}

/* Each GPU owner already holds the complete K(R). Its blocks can include
   atom pairs and translations outside the semilocal neighbor list. */
static void pack_exchange(const OpenMXExactExchangeOutputView *exchange,
                          OpenMXKPathBlock *blocks, double *values, int base_offset)
{
    const int spins = exchange->spin_channels;
    for (long b = 0; b < exchange->blocks; ++b) {
        const int a = exchange->block_atoms[2 * b], c = exchange->block_atoms[2 * b + 1];
        const int row = exchange->first_orbital[a], col = exchange->first_orbital[c];
        const int rows = exchange->first_orbital[a + 1] - row;
        const int cols = exchange->first_orbital[c + 1] - col;
        const int *cell = exchange->block_cell + 3 * b;
        const size_t source = exchange->block_offset[b];
        const int target = base_offset + (int)(source * spins);
        blocks[b] = (OpenMXKPathBlock){row, col, rows, cols, {cell[0], cell[1], cell[2]}, target};
        double *v = values + target;
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                for (int spin = 0; spin < spins; ++spin)
                    *v++ = exchange->values[spin][source + (size_t)i * cols + j];
    }
}
#endif

static int rank_counts(int ranks, const int *sizes, int field,
                       int *counts, int *offsets, int *total)
{
    *total = 0;
    for (int i = 0; i < ranks; ++i) {
        counts[i] = sizes[3 * i + field];
        offsets[i] = *total;
        if (counts[i] > INT_MAX - *total) return 0;
        *total += counts[i];
    }
    return 1;
}

static int band_file(const char *directory, const char *name, const double reciprocal[4][4],
                     const OpenMXChemicalPotentials *chemical, int path_count,
                     const int *counts, double ***paths, char ***names,
                     const OpenMXKPathMatrices *m, const int *output_to_unique,
                     const double *values)
{
    char file[YOUSO10];
    int order = m->noncollinear ? 2 * m->order : m->order;
    const int length = snprintf(file, sizeof(file), "%s%s%s", directory, name,
                                m->lno ? ".LNO.Band" : ".Band");
    if (length < 0 || length >= (int)sizeof(file)) return 0;
    FILE *fp = fopen(file, "w");
    if (!fp) return 0;

    fprintf(fp, " %d  %d  %18.15f\n", order, m->spin_channels - 1, chemical->chemical_potential);
    for (int i = 1; i <= 3; ++i)
        for (int j = 1; j <= 3; ++j)
            fprintf(fp, "%18.15f ", reciprocal[i][j]);
    fprintf(fp, "\n%d\n", path_count);
    for (int i = 1; i <= path_count; ++i) {
        fprintf(fp, "%d %18.15f %18.15f %18.15f  %18.15f %18.15f %18.15f  %s %s\n",
                counts[i], paths[i][1][1], paths[i][1][2], paths[i][1][3],
                paths[i][2][1], paths[i][2][2], paths[i][2][3], names[i][1], names[i][2]);
    }

    int index = 0;
    for (int path = 1; path <= path_count; ++path) {
        for (int point = 0; point < counts[path]; ++point) {
            double xyz[3];
            interpolate(paths[path], point, counts[path], xyz);
            for (int spin = 0; spin < m->spin_channels; ++spin) {
                const double *row = values + (output_to_unique[index] * m->spin_channels + spin) * (size_t)order;
                fprintf(fp, "%d %18.15f %18.15f %18.15f\n", order, xyz[0], xyz[1], xyz[2]);
                for (int i = 0; i < order; ++i) fprintf(fp, "%18.15f ", row[i]);
                fputc('\n', fp);
            }
            ++index;
        }
    }

    const int write_ok = !ferror(fp);
    return fclose(fp) == 0 && write_ok;
}

int openmx_band_kpath_try_gpu(const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    int spin, double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis, const struct OpenMXExactExchangeState *exact_exchange)
{
    if (scf_eigen_lib_flag != GPUSOLVER || !openmx_env_flag_at(OPENMX_ENV_GPU, 1)) return 0;
    if (OpenMX_HostSimdPreferred("Band_DFT_kpath")) return -1;

    const MPI_Comm comm = mpi_comm_level1;
    const KPathBasis basis = {atomnum, Matomnum, M2G, S_G2M, WhatSpecies, Spe_Total_CNO,
                             LNO_Num, FNAN, natn, ncn, atv_ijk};
    int rank, ranks, success = 0;
    const char *stage = "device selection";
    MPI_Comm devices = MPI_COMM_NULL;
    OpenMXKPathPoint *points = NULL;
    int *output_to_unique = NULL, *offset = NULL, *sizes = NULL, *counts = NULL;
    OpenMXKPathBlock *local = NULL, *blocks = NULL;
    double *local_values = NULL, *values = NULL, *local_eigen = NULL, *eigen = NULL;
    OpenMXKPathMatrices m = {0};
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const int device = device_owner(comm);

    m.noncollinear = spin == 3;
    m.spin_channels = spin == 1 ? 2 : 1;
    m.lno = local_basis != NULL;
    m.imaginary_diagonals = m.noncollinear &&
        (SO_switch || Hub_U_switch || Constraint_NCS_switch || Zeeman_NCS_switch || Zeeman_NCO_switch);
    m.overlap_cutoff = Threshold_OLP_Eigen;

    stage = "input preparation";
    int point_count = 0;
    for (int i = 1; i <= path_count; ++i) {
        if (points_per_path[i] < 0 || points_per_path[i] > INT_MAX - point_count) goto done;
        point_count += points_per_path[i];
    }
    if (point_count == 0) goto done;
    points = malloc((size_t)point_count * sizeof(*points));
    output_to_unique = malloc((size_t)point_count * sizeof(*output_to_unique));
    offset = malloc(((size_t)basis.atoms + 2) * sizeof(*offset));
    sizes = malloc((size_t)3 * ranks * sizeof(*sizes));
    counts = malloc((size_t)4 * ranks * sizeof(*counts));
    if (!all_ranks_ok(comm, points && output_to_unique && offset && sizes && counts)) goto done;
    const int unique_count = unique_points(path_count, points_per_path, paths, points, output_to_unique, point_count);
    m.order = orbital_offsets(&m, &basis, offset);
    const int multiplicity = m.noncollinear ? 2 : m.spin_channels;
    if (m.order <= 0 || unique_count <= 0 || m.order > INT_MAX / multiplicity ||
        unique_count > INT_MAX / (multiplicity * m.order)) goto done;
    const int bands_per_point = multiplicity * m.order;

    stage = "sparse matrix sizing";
    int local_size[3] = {0, 0, device >= 0};
    if (!all_ranks_ok(comm, local_blocks(&m, &basis, offset, hamiltonian, imaginary, overlap,
                                  local_basis, NULL, NULL, local_size))) goto done;
    MPI_Allgather(local_size, 3, MPI_INT, sizes, 3, MPI_INT, comm);

    int owner_count = 0, owner_index = 0;
    for (int i = 0; i < ranks; ++i) {
        if (i == rank) owner_index = owner_count;
        owner_count += sizes[3 * i + 2];
    }
    if (!owner_count) {
        stage = "device selection";
        goto done;
    }
    /* If there are more devices than points, only the first ones get work. */
    if (owner_count > unique_count) owner_count = unique_count;
    const int owner = device >= 0 && owner_index < owner_count;
    const int first = owner ? (int)((int64_t)owner_index * unique_count / owner_count) : 0;
    const int end = owner ? (int)((int64_t)(owner_index + 1) * unique_count / owner_count) : 0;

    int block_bytes, value_count;
    int *block_sizes = counts, *block_offsets = counts + ranks;
    int *value_sizes = counts + 2 * ranks, *value_offsets = counts + 3 * ranks;
    if (!rank_counts(ranks, sizes, 0, block_sizes, block_offsets, &block_bytes) ||
        !rank_counts(ranks, sizes, 1, value_sizes, value_offsets, &value_count) ||
        block_bytes == 0 || value_count == 0) goto done;
    const int semilocal_blocks = block_bytes / (int)sizeof(*blocks);
    m.block_count = semilocal_blocks;
    m.exchange_offset = m.value_count = value_count;
#if OPENMX_GPU_CUDA
    OpenMXExactExchangeOutputView exchange;
    const int exchange_status = Exact_Exchange_Output_Blocks(exact_exchange, &exchange);
    stage = "post-SCF exact exchange";
    if (!all_ranks_ok(comm, exchange_status == 0 ||
        (exchange_status == 1 && size_exchange(&m, &exchange, basis.atoms, offset)))) goto done;
#else
    (void)exact_exchange;
#endif

    // Host_ID is first in this communicator even if its GPU is unavailable.
    const int receiver = owner || rank == Host_ID;
    MPI_Comm_split(comm, receiver ? 0 : MPI_UNDEFINED,
                   rank == Host_ID ? 0 : rank + 1, &devices);
    const unsigned long long available = openmx_host_mem_available();
    const int recv_blocks = owner ? m.block_count : semilocal_blocks;
    const int recv_values = owner ? m.value_count : value_count;
    const size_t packed_bytes = (size_t)recv_blocks * sizeof(*blocks) + (size_t)recv_values * sizeof(*values);
    const size_t local_bytes = local_size[0] + (size_t)local_size[1] * sizeof(*values);
    stage = "host matrix allocation";
    /* A local guard accounts for the copies that can coexist on this node. */
    if (!all_ranks_ok(comm, !available || (local_bytes + (receiver ? packed_bytes : 0)) <= available / ranks))
        goto done;

    local = malloc(local_size[0] ? (size_t)local_size[0] : sizeof(*local));
    local_values = malloc((local_size[1] ? (size_t)local_size[1] : 1) * sizeof(*local_values));
    if (receiver) {
        blocks = malloc((size_t)recv_blocks * sizeof(*blocks));
        values = malloc((size_t)recv_values * sizeof(*values));
    }
    if (owner) local_eigen = malloc((size_t)(end - first) * bands_per_point * sizeof(*local_eigen));
    if (rank == Host_ID) eigen = malloc((size_t)unique_count * bands_per_point * sizeof(*eigen));
    if (!all_ranks_ok(comm, local && local_values && (!receiver || (blocks && values)) &&
                      (!owner || local_eigen) && (rank != Host_ID || eigen))) goto done;

    stage = "real-space matrix packing";
    if (!all_ranks_ok(comm, local_blocks(&m, &basis, offset, hamiltonian, imaginary, overlap,
                                  local_basis, local, local_values, local_size))) goto done;
    for (int i = 0; i < local_size[0] / (int)sizeof(*local); ++i)
        local[i].value_offset += value_offsets[rank];

    MPI_Gatherv(local, local_size[0], MPI_BYTE, blocks, block_sizes, block_offsets, MPI_BYTE,
                Host_ID, comm);
    MPI_Gatherv(local_values, local_size[1], MPI_DOUBLE, values, value_sizes, value_offsets, MPI_DOUBLE,
                Host_ID, comm);
    free(local);
    free(local_values);
    local = NULL;
    local_values = NULL;

    if (receiver) {
        MPI_Bcast(blocks, block_bytes, MPI_BYTE, 0, devices);
        MPI_Bcast(values, value_count, MPI_DOUBLE, 0, devices);
    }
#if OPENMX_GPU_CUDA
    if (owner && exchange_status == 1)
        pack_exchange(&exchange, blocks + semilocal_blocks, values, m.exchange_offset);
#endif
    m.blocks = blocks;
    m.values = values;

    stage = OPENMX_GPU_RUNTIME_NAME " matrix construction, transformation or eigensolve";
    if (rank == Host_ID && 0 < level_stdout) {
        printf("<Band_DFT_kpath> %s: %d k-points (%d distinct), %d GPU owner(s) with %s%s\n",
               m.lno ? ".LNO.Band" : ".Band", point_count, unique_count, owner_count,
               OPENMX_GPU_RUNTIME_NAME, m.exchange_offset < m.value_count ? " and exact exchange" : "");
        fflush(stdout);
    }
    if (!all_ranks_ok(comm, !owner || openmx_band_kpath_solve_gpu(device, &m, end - first,
                                                                points + first, local_eigen))) goto done;

    int index = 0;
    for (int i = 0; i < ranks; ++i) {
        const int has_work = sizes[3 * i + 2] && index < owner_count;
        const int begin = (int)((int64_t)index * unique_count / owner_count);
        index += has_work;
        const int finish = (int)((int64_t)index * unique_count / owner_count);
        value_sizes[i] = (finish - begin) * bands_per_point;
        value_offsets[i] = begin * bands_per_point;
    }
    MPI_Gatherv(local_eigen, (end - first) * bands_per_point, MPI_DOUBLE,
                eigen, value_sizes, value_offsets, MPI_DOUBLE, Host_ID, comm);
    stage = "file output";
    success = rank != Host_ID || band_file(filepath, filename, rtv, chemical, path_count,
                                           points_per_path, paths, names, &m, output_to_unique, eigen);
    success = all_ranks_ok(comm, success);

done:
    if (rank == Host_ID && 0 < level_stdout) {
        if (success) printf("<Band_DFT_kpath> %s (GPU-accelerated)\n", m.lno ? ".LNO.Band" : ".Band");
        else printf("<Band_DFT_kpath> %s unavailable; recomputing on CPU.\n", stage);
        fflush(stdout);
    }
    if (devices != MPI_COMM_NULL) MPI_Comm_free(&devices);
    free(eigen);
    free(local_eigen);
    free(values);
    free(blocks);
    free(local_values);
    free(local);
    free(counts);
    free(sizes);
    free(offset);
    free(output_to_unique);
    free(points);
    return success ? 1 : -1;
}

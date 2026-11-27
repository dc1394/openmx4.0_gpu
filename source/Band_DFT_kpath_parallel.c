/* MPI and legacy matrix views shared by the post-SCF band solvers. */
#include "openmx_common.h"
#include "openmx_band_kpath.h"
#include "openmx_band_kpath_points.h"
#include "openmx_host_memory.h"
#include "openmx_host_simd.h"
#include "openmx_env.h"
#include "openmx_exact_exchange.h"
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
#include "openmx_gpu_collective.h"
#include "openmx_gpu_device.h"
#include "openmx_mgpu_runtime.h"
#endif

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int atoms, local_atoms;
    const int *local_to_global, *global_to_halo, *species, *paos, *lnos, *neighbor_counts;
    int *const *neighbors, *const *cells, *const *translations;
} KPathBasis;

typedef struct { int blocks, values; } KPathLocalOffset;

typedef struct {
    MPI_Comm comm;
    int host, verbosity, simd;
    const char *directory, *name;
    const double (*reciprocal)[4];
} KPathOutput;

static int all_ranks_ok(MPI_Comm comm, int local)
{
    int ready;
    MPI_Allreduce(&local, &ready, 1, MPI_INT, MPI_MIN, comm);
    return ready;
}

/* Elect once on each node, including jobs where ranks see different
   ordinals for the same card. Prefer the first usable rank on each GPU. */
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
static int device_owner(MPI_Comm comm)
{
    unsigned char uuid[16];
    int device = -1, first;
    const int usable = gpu_rank_device_usable() && OpenMXGpuGetDevice(&device) == OPENMX_GPU_SUCCESS &&
                       openmx_gpu_physical_uuid(device, uuid);
    (void)openmx_gpu_local_sharers(comm, usable ? uuid : NULL, &first);
    return first ? device : -1;
}
#endif

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
    const size_t entry_stride = m->planar ? 1 : (size_t)values_per_orbital_pair(m);
    const size_t channel_stride = m->planar ? (size_t)rows * cols : 1;
    double **overlap = s[atom][neighbor];
    if (m->lno) {
        const int a = basis->local_to_global[atom], b = basis->neighbors[a][neighbor];
        const int local_b = basis->global_to_halo[b];
        const int paos_a = basis->paos[basis->species[a]], paos_b = basis->paos[basis->species[b]];
        for (int spin = 0; spin < m->spin_channels; ++spin) {
            const double *left_rows = openmx_local_lno_row(local_basis, spin, atom);
            const double *right_rows = openmx_local_lno_row(local_basis, spin, local_b);
            double **block = h[spin][atom][neighbor];
            for (int i = 0; i < rows; ++i) {
                const double *left = left_rows + (size_t)i * paos_a;
                for (int j = 0; j < cols; ++j) {
                    const double *right = right_rows + (size_t)j * paos_b;
                    double *v = values + ((size_t)j * rows + i) * entry_stride + 2 * spin * channel_stride;
                    v[0] = lno_project(left, overlap, right, paos_a, paos_b);
                    v[channel_stride] = lno_project(left, block, right, paos_a, paos_b);
                }
            }
        }
        return;
    }

    double **up = h[0][atom][neighbor];
    if (!m->noncollinear) {
        double **down = m->spin_channels == 2 ? h[1][atom][neighbor] : NULL;
        for (int i = 0; i < rows; ++i) {
            const double *sr = overlap[i], *ur = up[i], *dr = down ? down[i] : NULL;
            for (int j = 0; j < cols; ++j) {
                double *v = values + ((size_t)j * rows + i) * entry_stride;
                v[0] = sr[j];
                v[channel_stride] = ur[j];
                if (dr) v[2 * channel_stride] = dr[j];
            }
        }
        return;
    }

    double **down = h[1][atom][neighbor], **mix = h[2][atom][neighbor], **imag_mix = h[3][atom][neighbor];
    double **imag_up = m->imaginary_diagonals ? ih[0][atom][neighbor] : NULL;
    double **imag_down = m->imaginary_diagonals ? ih[1][atom][neighbor] : NULL;
    double **imag_extra = m->imaginary_diagonals ? ih[2][atom][neighbor] : NULL;
    for (int i = 0; i < rows; ++i) {
        const double *sr = overlap[i], *ur = up[i], *dr = down[i], *mr = mix[i], *ir = imag_mix[i];
        const double *iu = imag_up ? imag_up[i] : NULL, *id = imag_down ? imag_down[i] : NULL;
        const double *ie = imag_extra ? imag_extra[i] : NULL;
        for (int j = 0; j < cols; ++j) {
            double *v = values + ((size_t)j * rows + i) * entry_stride;
            v[0] = sr[j];
            v[channel_stride] = ur[j];
            v[2 * channel_stride] = dr[j];
            v[3 * channel_stride] = mr[j];
            v[4 * channel_stride] = ir[j] + (ie ? ie[j] : 0.0);
            if (ie) { v[5 * channel_stride] = iu[j]; v[6 * channel_stride] = id[j]; }
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

/* Per-atom prefixes let threads fill disjoint spans in the original order. */
static int local_block_offsets(const OpenMXKPathMatrices *m, const KPathBasis *basis, const int *offset,
                               KPathLocalOffset *prefix)
{
    int count = 0, value_count = 0, width = values_per_orbital_pair(m);
    for (int atom = 1; atom <= basis->local_atoms; ++atom) {
        prefix[atom - 1] = (KPathLocalOffset){count, value_count};
        const int a = basis->local_to_global[atom], rows = offset[a + 1] - offset[a];
        for (int neighbor = 0; neighbor <= basis->neighbor_counts[a]; ++neighbor) {
            const int b = basis->neighbors[a][neighbor], cols = offset[b + 1] - offset[b];
            if (rows == 0 || cols == 0) continue;
            if (count >= INT_MAX / (int)sizeof(OpenMXKPathBlock) ||
                (size_t)rows * cols > (size_t)(INT_MAX - value_count) / width) return 0;
            ++count;
            value_count += rows * cols * width;
        }
    }
    prefix[basis->local_atoms] = (KPathLocalOffset){count, value_count};
    return 1;
}

static void pack_local_blocks(const OpenMXKPathMatrices *m, const KPathBasis *basis, const int *offset,
    const KPathLocalOffset *prefix, double *****h, double *****ih, double ****s,
    const OpenMXLocalBasisViews *local_basis, OpenMXKPathBlock *blocks, double *values, int base_offset)
{
    const int width = values_per_orbital_pair(m);
#pragma omp parallel for schedule(static) if (prefix[basis->local_atoms].values >= 65536 && omp_get_max_threads() > 1)
    for (int atom = 1; atom <= basis->local_atoms; ++atom) {
        const int a = basis->local_to_global[atom], rows = offset[a + 1] - offset[a];
        int block = prefix[atom - 1].blocks, value = prefix[atom - 1].values;
        for (int neighbor = 0; neighbor <= basis->neighbor_counts[a]; ++neighbor) {
            const int b = basis->neighbors[a][neighbor], cols = offset[b + 1] - offset[b];
            if (rows == 0 || cols == 0) continue;
            const int *shift = basis->translations[basis->cells[a][neighbor]];
            blocks[block++] = (OpenMXKPathBlock){
                offset[a], offset[b], rows, cols, {shift[1], shift[2], shift[3]}, base_offset + value
            };
            pack_values(m, basis, atom, neighbor, rows, cols, h, ih, s, local_basis, values + value);
            value += rows * cols * width;
        }
    }
}

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

/* Each solver rank already holds the complete K(R). Its blocks can include
   atom pairs and translations outside the semilocal neighbor list. */
static void pack_exchange(const OpenMXExactExchangeOutputView *exchange,
                          OpenMXKPathBlock *blocks, double *values, int base_offset, int planar)
{
    const int spins = exchange->spin_channels;
#pragma omp parallel for schedule(static) if (exchange->block_offset[exchange->blocks] >= 65536 && omp_get_max_threads() > 1)
    for (long b = 0; b < exchange->blocks; ++b) {
        const int a = exchange->block_atoms[2 * b], c = exchange->block_atoms[2 * b + 1];
        const int row = exchange->first_orbital[a], col = exchange->first_orbital[c];
        const int rows = exchange->first_orbital[a + 1] - row;
        const int cols = exchange->first_orbital[c + 1] - col;
        const int *cell = exchange->block_cell + 3 * b;
        const size_t source = exchange->block_offset[b];
        const int target = base_offset + (int)(source * spins);
        const size_t entry_stride = planar ? 1 : (size_t)spins;
        const size_t channel_stride = planar ? (size_t)rows * cols : 1;
        blocks[b] = (OpenMXKPathBlock){row, col, rows, cols, {cell[0], cell[1], cell[2]}, target};
        for (int spin = 0; spin < spins; ++spin) {
            const double *input = exchange->values[spin] + source;
            double *output = values + target + spin * channel_stride;
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    output[((size_t)j * rows + i) * entry_stride] = input[(size_t)i * cols + j];
        }
    }
}

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

static int run_parallel(const KPathOutput *output, const KPathBasis *basis, OpenMXKPathMatrices m,
    int use_gpu, const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis, const struct OpenMXExactExchangeState *exact_exchange)
{
    const MPI_Comm comm = output->comm;
    int rank, ranks, success = 0;
    int device = -1, threads_used = 0;
    const char *stage = "input preparation";
    MPI_Comm owners = MPI_COMM_NULL;
    OpenMXKPathPoint *points = NULL;
    int *output_to_unique = NULL, *offset = NULL, *sizes = NULL, *counts = NULL;
    KPathLocalOffset *local_offsets = NULL;
    OpenMXKPathBlock *local = NULL, *blocks = NULL;
    double *local_values = NULL, *values = NULL, *local_eigen = NULL, *eigen = NULL;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
    if (use_gpu) device = device_owner(comm);
#endif
    m.planar = !use_gpu;

    int point_count = 0;
    for (int i = 1; i <= path_count; ++i) {
        if (points_per_path[i] < 0 || points_per_path[i] > INT_MAX - point_count) goto done;
        point_count += points_per_path[i];
    }
    if (point_count == 0) goto done;
    points = malloc((size_t)point_count * sizeof(*points));
    output_to_unique = malloc((size_t)point_count * sizeof(*output_to_unique));
    offset = malloc(((size_t)basis->atoms + 2) * sizeof(*offset));
    sizes = malloc((size_t)3 * ranks * sizeof(*sizes));
    counts = malloc((size_t)4 * ranks * sizeof(*counts));
    local_offsets = malloc(((size_t)basis->local_atoms + 1) * sizeof(*local_offsets));
    if (!all_ranks_ok(comm, points && output_to_unique && offset && sizes && counts && local_offsets)) goto done;
    const int unique_count = unique_points(path_count, points_per_path, paths, points, output_to_unique, point_count);
    m.order = orbital_offsets(&m, basis, offset);
    const int multiplicity = m.noncollinear ? 2 : m.spin_channels;
    if (m.order <= 0 || unique_count <= 0 || m.order > INT_MAX / multiplicity ||
        unique_count > INT_MAX / (multiplicity * m.order)) goto done;
    const int bands_per_point = multiplicity * m.order;

    stage = "sparse matrix sizing";
    if (!all_ranks_ok(comm, local_block_offsets(&m, basis, offset, local_offsets))) goto done;
    const KPathLocalOffset end_offset = local_offsets[basis->local_atoms];
    const int eligible = use_gpu ? device >= 0 : 1;
    int local_size[3] = {end_offset.blocks * (int)sizeof(*local), end_offset.values, eligible};
    MPI_Allgather(local_size, 3, MPI_INT, sizes, 3, MPI_INT, comm);

    int owner_count = 0, owner_index = 0, input_root = -1;
    for (int i = 0; i < ranks; ++i) {
        if (i == rank) owner_index = owner_count;
        if (owner_count == 0 && sizes[3 * i + 2]) input_root = i;
        owner_count += sizes[3 * i + 2];
    }
    if (!owner_count) {
        stage = "device selection";
        goto done;
    }
    /* If there are more eligible ranks than points, only the first ones solve. */
    if (owner_count > unique_count) owner_count = unique_count;
    const int owner = eligible && owner_index < owner_count;
    const int gather_root = rank == input_root;
    const int replicate = !use_gpu && owner_count == ranks;
    const int direct_pack = replicate || gather_root;
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
    OpenMXExactExchangeOutputView exchange;
    const int exchange_status = Exact_Exchange_Output_Blocks(exact_exchange, &exchange);
    stage = "post-SCF exact exchange";
    if (!all_ranks_ok(comm, exchange_status == 0 ||
        (exchange_status == 1 && size_exchange(&m, &exchange, basis->atoms, offset)))) goto done;

    /* When only a subset is solving, share the payload only with those ranks. */
    if (!replicate && owner_count > 1) MPI_Comm_split(comm, owner ? 0 : MPI_UNDEFINED, rank, &owners);
    int local_ranks = ranks;
#ifndef nompi
    if (!use_gpu) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_size(node, &local_ranks);
        MPI_Comm_free(&node);
    }
#endif
    const unsigned long long available = openmx_host_mem_available();
    const size_t packed_bytes = (size_t)m.block_count * sizeof(*blocks) + (size_t)m.value_count * sizeof(*values);
    const size_t local_bytes = local_size[0] + (size_t)local_size[1] * sizeof(*values);
    stage = "host matrix allocation";
    /* A local guard accounts for the copies that can coexist on this node. */
    const size_t eigen_bytes = ((owner ? (size_t)(end - first) : 0) +
                               (rank == output->host ? (size_t)unique_count : 0)) * bands_per_point * sizeof(*eigen);
    const size_t host_bytes = (direct_pack ? 0 : local_bytes) + (owner ? packed_bytes : 0) + eigen_bytes;
    const size_t budget = available ? available / local_ranks : SIZE_MAX;
    if (!all_ranks_ok(comm, host_bytes <= budget))
        goto done;

    if (!direct_pack) {
        local = malloc(local_size[0] ? (size_t)local_size[0] : sizeof(*local));
        local_values = malloc((local_size[1] ? (size_t)local_size[1] : 1) * sizeof(*local_values));
    }
    if (owner) {
        blocks = malloc((size_t)m.block_count * sizeof(*blocks));
        values = malloc((size_t)m.value_count * sizeof(*values));
        local_eigen = malloc((size_t)(end - first) * bands_per_point * sizeof(*local_eigen));
    }
    if (rank == output->host) eigen = malloc((size_t)unique_count * bands_per_point * sizeof(*eigen));
    if (!all_ranks_ok(comm, (direct_pack || (local && local_values)) && (!owner || (blocks && values && local_eigen)) &&
                      (rank != output->host || eigen))) goto done;

    stage = "real-space matrix packing";
    OpenMXKPathBlock *local_block_target = direct_pack ? blocks + block_offsets[rank] / sizeof(*blocks) : local;
    double *local_value_target = direct_pack ? values + value_offsets[rank] : local_values;
    pack_local_blocks(&m, basis, offset, local_offsets, hamiltonian, imaginary, overlap, local_basis,
                      local_block_target, local_value_target, value_offsets[rank]);

    MPI_Request gather[2];
    int mpi_status;
    if (replicate) {
        mpi_status = MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                                    blocks, block_sizes, block_offsets, MPI_BYTE, comm, &gather[0]);
        if (mpi_status == MPI_SUCCESS)
            mpi_status = MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                                        values, value_sizes, value_offsets, MPI_DOUBLE, comm, &gather[1]);
    } else {
        mpi_status = MPI_Igatherv(gather_root ? MPI_IN_PLACE : local, local_size[0], MPI_BYTE,
                                  blocks, block_sizes, block_offsets, MPI_BYTE, input_root, comm, &gather[0]);
        if (mpi_status == MPI_SUCCESS)
            mpi_status = MPI_Igatherv(gather_root ? MPI_IN_PLACE : local_values, local_size[1], MPI_DOUBLE,
                                      values, value_sizes, value_offsets, MPI_DOUBLE, input_root, comm, &gather[1]);
    }
    if (mpi_status != MPI_SUCCESS) {
        MPI_Abort(comm, EXIT_FAILURE);
        abort();
    }
    /* MPI fills the semilocal prefix while each owner packs its own K(R). */
    if (owner && exchange_status == 1)
        pack_exchange(&exchange, blocks + semilocal_blocks, values, m.exchange_offset, m.planar);
    if (MPI_Waitall(2, gather, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        MPI_Abort(comm, EXIT_FAILURE);
        abort();
    }
    free(local);
    free(local_values);
    local = NULL;
    local_values = NULL;

    if (owners != MPI_COMM_NULL) {
        MPI_Bcast(blocks, block_bytes, MPI_BYTE, 0, owners);
        MPI_Bcast(values, value_count, MPI_DOUBLE, 0, owners);
    }
    m.blocks = blocks;
    m.values = values;

    const char *backend = use_gpu ? OPENMX_GPU_RUNTIME_NAME : openmx_core_host_simd_name(output->simd);
    stage = "matrix construction, transformation or eigensolve";
    if (rank == output->host && 0 < output->verbosity) {
        printf("<Band_DFT_kpath> %s: %d k-points (%d distinct), %d %s with %s%s\n",
               m.lno ? ".LNO.Band" : ".Band", point_count, unique_count, owner_count, use_gpu ? "GPU owner(s)" : "CPU rank(s)",
               backend, m.exchange_offset < m.value_count ? " and exact exchange" : "");
        fflush(stdout);
    }
    int solved = 1;
    if (owner) {
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
        if (use_gpu) solved = openmx_band_kpath_solve_gpu(device, &m, end - first, points + first, local_eigen);
        else
#endif
            solved = openmx_band_kpath_solve_cpu(&m, end - first, points + first, local_eigen,
                                                output->simd, budget - host_bytes, &threads_used);
    }
    if (!all_ranks_ok(comm, solved)) goto done;

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
                eigen, value_sizes, value_offsets, MPI_DOUBLE, output->host, comm);
    stage = "file output";
    success = rank != output->host || band_file(output->directory, output->name, output->reciprocal,
        chemical, path_count, points_per_path, paths, names, &m, output_to_unique, eigen);
    success = all_ranks_ok(comm, success);

done:
    if (rank == output->host && 0 < output->verbosity) {
        if (success && use_gpu) printf("<Band_DFT_kpath> %s (GPU-accelerated)\n", m.lno ? ".LNO.Band" : ".Band");
        else if (success) printf("<Band_DFT_kpath> %s (%d OpenMP worker(s) on rank %d)\n",
                                 m.lno ? ".LNO.Band" : ".Band", threads_used, rank);
        else printf("<Band_DFT_kpath> %s %s unavailable.\n", use_gpu ? "GPU" : "CPU", stage);
        fflush(stdout);
    }
    if (owners != MPI_COMM_NULL) MPI_Comm_free(&owners);
    free(eigen);
    free(local_eigen);
    free(values);
    free(blocks);
    free(local_values);
    free(local);
    free(local_offsets);
    free(counts);
    free(sizes);
    free(offset);
    free(output_to_unique);
    free(points);
    return success;
}

int openmx_band_kpath_parallel(const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    int spin, double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis, const struct OpenMXExactExchangeState *exact_exchange)
{
    const KPathBasis basis = {atomnum, Matomnum, M2G, S_G2M, WhatSpecies, Spe_Total_CNO,
                             LNO_Num, FNAN, natn, ncn, atv_ijk};
    const KPathOutput output = {mpi_comm_level1, Host_ID, level_stdout, OpenMX_HostSimdMode(),
                                filepath, filename, rtv};
    OpenMXKPathMatrices m = {0};
    m.noncollinear = spin == 3;
    m.spin_channels = spin == 1 ? 2 : 1;
    m.lno = local_basis != NULL;
    m.imaginary_diagonals = m.noncollinear &&
        (SO_switch || Hub_U_switch || Constraint_NCS_switch || Zeeman_NCS_switch || Zeeman_NCO_switch);
    m.overlap_cutoff = Threshold_OLP_Eigen;

#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
    const int gpu = scf_eigen_lib_flag == GPUSOLVER && openmx_env_flag_at(OPENMX_ENV_GPU, 1) &&
                    !OpenMX_HostSimdPreferred("Band_DFT_kpath");
    if (all_ranks_ok(output.comm, gpu) &&
        run_parallel(&output, &basis, m, 1, chemical, path_count, points_per_path, paths, names,
                     hamiltonian, imaginary, overlap, local_basis, exact_exchange)) return 1;
#endif
    return all_ranks_ok(output.comm, output.simd != OPENMX_HOST_SIMD_LEGACY) &&
           run_parallel(&output, &basis, m, 0, chemical, path_count, points_per_path, paths, names,
                        hamiltonian, imaginary, overlap, local_basis, exact_exchange);
}

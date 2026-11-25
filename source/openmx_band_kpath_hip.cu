#include "openmx_band_kpath_gpu.h"
#include "openmx_gpu_memory.hpp"
#include "openmx_mgpu_runtime.h"
#include "openmx_stdout_level.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <new>
#include <vector>

extern "C" {
OpenMXBlasStatus openmx_gemmul8Zgemm(OpenMXBlasHandle, OpenMXBlasOperation, OpenMXBlasOperation,
    int, int, int, const OpenMXGpuComplex *, const OpenMXGpuComplex *, int,
    const OpenMXGpuComplex *, int, const OpenMXGpuComplex *, OpenMXGpuComplex *, int);
void openmx_gemmul8TrimWorkspaces(void);
int openmx_magma_zheevdx_gpu(int, int, void *, double *, int *);
int openmx_hipsolver_zheevd_gpu(int, int, void *, double *, int *);
int openmx_gpu_eigen_backend(int, int, int);
void openmx_magma_ozaki2_release(void);
int openmx_magma_release_z_workspace(void);
int openmx_hipsolver_release_workspace(void);
}

namespace {
using Z = OpenMXGpuComplex;
using openmx::gpu::DeviceBuffer;
constexpr int threads = 128;
constexpr double two_pi = 6.283185307179586476925;

struct BlockGroup {
    int first, end;
};

__global__ void bloch_phases(int count, const OpenMXKPathBlock *blocks, OpenMXKPathPoint point, Z *phases)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int *t = blocks[i].translation;
        const double kr = point.xyz[0] * t[0] + point.xyz[1] * t[1] + point.xyz[2] * t[2];
        double sine, cosine;
        sincos(two_pi * kr, &sine, &cosine);
        phases[i] = OpenMXGpuMakeComplex(cosine, sine);
    }
}

__device__ void add_bloch(Z &sum, Z phase, double real, double imag)
{
    sum.x += phase.x * real - phase.y * imag;
    sum.y += phase.y * real + phase.x * imag;
}

/* Each group owns a dense atom-pair block. Sum periodic images in the
   original order, with one thread per orbital pair and no atomic adds. */
__global__ void assemble(int n, int matrix_order, int width, int spin, bool lno,
                         bool noncollinear, bool imaginary_diagonals,
                         const BlockGroup *groups, const OpenMXKPathBlock *blocks,
                         const double *values, const Z *phases, Z *s, Z *h)
{
    const BlockGroup group = groups[blockIdx.x];
    const OpenMXKPathBlock pair = blocks[group.first];
    for (int entry = threadIdx.x; entry < pair.rows * pair.cols; entry += blockDim.x) {
        Z overlap{}, up{}, down{}, mix{};
        for (int image = group.first; image < group.end; ++image) {
            const double *v = values + blocks[image].value_offset + (size_t)entry * width;
            const Z phase = phases[image];
            if (noncollinear) {
                add_bloch(overlap, phase, v[0], 0.0);
                add_bloch(up, phase, v[1], imaginary_diagonals ? v[5] : 0.0);
                add_bloch(down, phase, v[2], imaginary_diagonals ? v[6] : 0.0);
                add_bloch(mix, phase, v[3], v[4]);
            } else {
                if (s) add_bloch(overlap, phase, v[lno ? 2 * spin : 0], 0.0);
                add_bloch(up, phase, v[lno ? 2 * spin + 1 : spin + 1], 0.0);
            }
        }

        const int row = pair.row + entry % pair.rows;
        const int col = pair.col + entry / pair.rows;
        if (s) s[row + (size_t)col * n] = overlap;
        h[row + (size_t)col * matrix_order] = up;
        if (noncollinear) {
            h[row + n + (size_t)(col + n) * matrix_order] = down;
            h[row + (size_t)(col + n) * matrix_order] = mix;
            h[col + n + (size_t)row * matrix_order] = OpenMXGpuMakeComplex(mix.x, -mix.y);
        }
    }
}

__global__ void scale_overlap(int n, Z *vectors, const double *values)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n * n) {
        const double factor = 1.0 / sqrt(values[i / n]);
        vectors[i].x *= factor;
        vectors[i].y *= factor;
    }
}

/* In the CPU noncollinear ordering, the two spin entries of each basis
   vector are adjacent. Here they are n entries apart. The cutoff decision
   still comes from the first spin diagonal and applies to both entries. */
__global__ void penalty_diagonal(int n, int order, double cutoff, const double *values,
                                 Z *h, unsigned char *discard)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t stride = (size_t)order + 1;
        const double penalty = values[i] < cutoff ? pow(values[i] / cutoff, -2.0) - 1.0 : 0.0;
        h[i * stride].x += penalty;
        if (order != n) h[(i + n) * stride].x += penalty;
        discard[i] = h[i * stride].x > 1.0e3;
    }
}

__global__ void remove_penalized(int n, int order, const unsigned char *discard, Z *h)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)order * order) {
        const int row = i % order, col = i / order;
        if (discard[row % n] || discard[col % n])
            h[i] = OpenMXGpuMakeComplex(row == col ? 1.0e4 : 0.0, 0.0);
    }
}

bool good_spectrum(int n, const double *values)
{
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(values[i]) || (i > 0 && values[i] < values[i - 1])) return false;
    return true;
}

/* MAGMA's small-matrix driver uses LAPACK. Use hipSOLVER for these
   matrices and for eigenvalues-only requests under the rocSOLVER policy. */
struct HipEigensolver {
    OpenMXSolverHandle handle = nullptr;
    DeviceBuffer<Z> work;
    DeviceBuffer<double> eigenvalues;
    DeviceBuffer<int> info;
    int previous_order = 0, work_size = 0;

    ~HipEigensolver()
    {
        if (handle) (void)OpenMXSolverDestroy(handle);
    }

    bool solve(int order, Z *matrix, double *values)
    {
        const auto job = HIPSOLVER_EIG_MODE_NOVECTOR;
        if (!handle && OpenMXSolverCreate(&handle) != OPENMX_SOLVER_SUCCESS) return false;
        if (!info && !info.allocate(1)) return false;
        if (eigenvalues.size() < (size_t)order) {
            eigenvalues.reset();
            if (!eigenvalues.allocate(order)) return false;
        }
        if (previous_order != order) {
            if (hipsolverDnZheevd_bufferSize(handle, job, OPENMX_BLAS_FILL_LOWER, order, matrix,
                                            order, eigenvalues.get(), &work_size) != OPENMX_SOLVER_SUCCESS ||
                work_size < 0) return false;
            if (work.size() < (size_t)work_size) {
                work.reset();
                if (!work.allocate(work_size)) return false;
            }
            previous_order = order;
        }
        if (hipsolverDnZheevd(handle, job, OPENMX_BLAS_FILL_LOWER, order, matrix, order,
                             eigenvalues.get(), work.get(), work_size, info.get()) != OPENMX_SOLVER_SUCCESS)
            return false;

        int result = 0;
        return OpenMXGpuMemcpy(&result, info.get(), sizeof(result), OPENMX_GPU_COPY_DEVICE_TO_HOST) ==
                   OPENMX_GPU_SUCCESS && result == 0 &&
               OpenMXGpuMemcpy(values, eigenvalues.get(), (size_t)order * sizeof(*values),
                               OPENMX_GPU_COPY_DEVICE_TO_HOST) == OPENMX_GPU_SUCCESS;
    }
};

/* Native buffers remain resident for all k-points assigned to this GPU. */
struct Workspace {
    DeviceBuffer<OpenMXKPathBlock> blocks;
    DeviceBuffer<BlockGroup> groups;
    DeviceBuffer<double> values, overlap_values;
    DeviceBuffer<Z> phases, s, h, product;
    DeviceBuffer<unsigned char> discard;
    OpenMXBlasHandle blas = nullptr;
    HipEigensolver solver;
    int group_count = 0;

    ~Workspace()
    {
        if (blas) (void)OpenMXBlasDestroy(blas);
    }

    bool prepare(const OpenMXKPathMatrices &m, int order)
    {
        const size_t n2 = (size_t)m.order * m.order, order2 = (size_t)order * order;
        if (order2 > std::numeric_limits<int>::max() / 4) return false;

        size_t free = 0, total = 0;
        const size_t data = (size_t)m.value_count * sizeof(double) +
                            (size_t)m.block_count * (sizeof(OpenMXKPathBlock) + sizeof(BlockGroup) + sizeof(Z));
        const size_t dense = (n2 + 2 * order2) * sizeof(Z);
        const size_t solver_room = 2 * order2 * sizeof(Z);
        constexpr size_t reserve = size_t{512} << 20;
        if (OpenMXGpuMemGetInfo(&free, &total) != OPENMX_GPU_SUCCESS ||
            free < data + dense + solver_room + reserve) return false;

        std::vector<OpenMXKPathBlock> ordered(m.blocks, m.blocks + m.block_count);
        std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
            if (a.row != b.row) return a.row < b.row;
            if (a.col != b.col) return a.col < b.col;
            return a.value_offset < b.value_offset;
        });
        std::vector<BlockGroup> spans;
        for (int i = 0; i < m.block_count;) {
            const int first = i++;
            while (i < m.block_count && ordered[i].row == ordered[first].row &&
                   ordered[i].col == ordered[first].col) ++i;
            spans.push_back({first, i});
        }
        group_count = (int)spans.size();

        return blocks.allocate(m.block_count) && groups.allocate(group_count) &&
               values.allocate(m.value_count) && phases.allocate(m.block_count) &&
               s.allocate(n2) && h.allocate(order2) && product.allocate(order2) &&
               overlap_values.allocate(m.order) && discard.allocate(m.order) &&
               OpenMXBlasCreate(&blas) == OPENMX_BLAS_SUCCESS &&
               OpenMXGpuMemcpy(blocks.get(), ordered.data(), blocks.size() * sizeof(ordered[0]),
                               OPENMX_GPU_COPY_HOST_TO_DEVICE) == OPENMX_GPU_SUCCESS &&
               OpenMXGpuMemcpy(groups.get(), spans.data(), groups.size() * sizeof(spans[0]),
                               OPENMX_GPU_COPY_HOST_TO_DEVICE) == OPENMX_GPU_SUCCESS &&
               OpenMXGpuMemcpy(values.get(), m.values, values.size() * sizeof(double),
                               OPENMX_GPU_COPY_HOST_TO_DEVICE) == OPENMX_GPU_SUCCESS;
    }

    bool construct(const OpenMXKPathMatrices &m, int order, int spin, bool build_overlap)
    {
        if (OpenMXGpuMemset(h.get(), 0, h.size() * sizeof(Z)) != OPENMX_GPU_SUCCESS ||
            (build_overlap && OpenMXGpuMemset(s.get(), 0, s.size() * sizeof(Z)) != OPENMX_GPU_SUCCESS))
            return false;
        const int width = m.noncollinear ? (m.imaginary_diagonals ? 7 : 5) :
                          (m.lno ? 2 * m.spin_channels : 1 + m.spin_channels);
        assemble<<<group_count, threads>>>(m.order, order, width, spin, m.lno, m.noncollinear,
            m.imaginary_diagonals, groups.get(), blocks.get(), values.get(), phases.get(),
            build_overlap ? s.get() : nullptr, h.get());
        return OpenMXGpuGetLastError() == OPENMX_GPU_SUCCESS;
    }

    bool diagonalize(int order, Z *matrix, double *eigen, bool vectors)
    {
        if (OpenMXGpuDeviceSynchronize() != OPENMX_GPU_SUCCESS) return false;
        int found = order, status;
        if (order <= 128 || openmx_gpu_eigen_backend(order, order, 1)) {
            if (!vectors) return solver.solve(order, matrix, eigen) && good_spectrum(order, eigen);
            status = openmx_hipsolver_zheevd_gpu(order, order, matrix, eigen, &found);
        } else {
            status = vectors ? openmx_magma_zheevdx_gpu(order, order, matrix, eigen, &found)
                             : openmx_magma_zheev_values_gpu(order, matrix, eigen);
            // MAGMA destroys its solve streams. Release their Ozaki-II
            // scratch before the next solve creates new streams.
            openmx_magma_ozaki2_release();
        }
        return status == 0 && found == order && good_spectrum(order, eigen);
    }

    bool orthogonalize(int n, double *eigen)
    {
        if (!diagonalize(n, s.get(), eigen, true)) return false;
        for (int i = 0; i < n; ++i) {
            if (eigen[i] < 0.0) eigen[i] = 1.0e-14;
            if (eigen[i] == 0.0) return false;
        }
        if (OpenMXGpuMemcpy(overlap_values.get(), eigen, (size_t)n * sizeof(*eigen),
                            OPENMX_GPU_COPY_HOST_TO_DEVICE) != OPENMX_GPU_SUCCESS) return false;
        scale_overlap<<<(s.size() + threads - 1) / threads, threads>>>(n, s.get(), overlap_values.get());
        return OpenMXGpuGetLastError() == OPENMX_GPU_SUCCESS;
    }

    bool transform(int n, int order, bool native_precision)
    {
        const auto zgemm = native_precision ? OpenMXBlasZgemm : openmx_gemmul8Zgemm;
        const Z one = OpenMXGpuMakeComplex(1.0, 0.0), zero = OpenMXGpuMakeComplex(0.0, 0.0);
        // U is block diagonal for the two spin components. Keep its n-by-n
        // storage and apply it to the two halves without zero products.
        for (int col = 0; col < order; col += n) {
            const size_t start = (size_t)col * order;
            if (zgemm(blas, OPENMX_BLAS_OP_N, OPENMX_BLAS_OP_N, order, n, n, &one,
                      h.get() + start, order, s.get(), n, &zero, product.get() + start, order) !=
                OPENMX_BLAS_SUCCESS) return false;
        }
        for (int row = 0; row < order; row += n) {
            if (zgemm(blas, OPENMX_BLAS_OP_C, OPENMX_BLAS_OP_N, n, order, n, &one,
                      s.get(), n, product.get() + row, order, &zero, h.get() + row, order) !=
                OPENMX_BLAS_SUCCESS) return false;
        }
        return true;
    }
};

bool solve_points(Workspace &w, const OpenMXKPathMatrices &m, int point_count,
                  const OpenMXKPathPoint *points, double *eigen, const char **stage)
{
    const int order = m.noncollinear ? 2 * m.order : m.order;
    const int n = m.order;
    *stage = "device workspace allocation";
    if (!w.prepare(m, order)) return false;

    std::vector<double> overlap(n);
    for (int k = 0; k < point_count; ++k) {
        *stage = "Bloch phases";
        bloch_phases<<<(m.block_count + threads - 1) / threads, threads>>>(
            m.block_count, w.blocks.get(), points[k], w.phases.get());
        if (OpenMXGpuGetLastError() != OPENMX_GPU_SUCCESS) return false;

        for (int spin = 0; spin < m.spin_channels; ++spin) {
            const bool overlap_needed = spin == 0 || m.lno;
            *stage = "Bloch matrix assembly";
            if (!w.construct(m, order, spin, overlap_needed)) return false;
            *stage = "overlap eigensolve and scaling";
            if (overlap_needed && !w.orthogonalize(n, overlap.data())) return false;

            *stage = "Hamiltonian transformation";
            const bool ill_conditioned = *std::min_element(overlap.begin(), overlap.end()) < m.overlap_cutoff;
            if (!w.transform(n, order, ill_conditioned)) return false;
            if (!m.lno) {
                penalty_diagonal<<<(n + threads - 1) / threads, threads>>>(
                    n, order, m.overlap_cutoff, w.overlap_values.get(), w.h.get(), w.discard.get());
                if (OpenMXGpuGetLastError() != OPENMX_GPU_SUCCESS) return false;
                remove_penalized<<<(w.h.size() + threads - 1) / threads, threads>>>(
                    n, order, w.discard.get(), w.h.get());
                if (OpenMXGpuGetLastError() != OPENMX_GPU_SUCCESS) return false;
            }

            *stage = "Hamiltonian eigenvalues";
            double *row = eigen + ((size_t)k * m.spin_channels + spin) * order;
            if (!w.diagonalize(order, w.h.get(), row, false)) return false;
        }
        if (OPENMX_STDOUT_LEVEL(2) && ((k + 1) % 10 == 0 || k + 1 == point_count)) {
            std::printf("<Band_DFT_kpath> GPU owner completed %d/%d distinct k-points\n", k + 1, point_count);
            std::fflush(stdout);
        }
    }
    return true;
}
} // namespace

extern "C" int openmx_band_kpath_solve_hip(int device, const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues)
{
    openmx::gpu::DeviceBinding binding(device);
    if (!binding) return 0;
    // Products and eigensolves stay on this owner, independent of any
    // previously requested cooperative SCF operation.
    int previous = openmx_mgpu_scope_enter(0);
    void *admission = openmx_mgpu_admit_devices(&device, 1);
    bool success = false;
    const char *stage = "device admission";
    if (admission) {
        openmx_magma_ozaki2_release();
        Workspace work;
        try {
            success = solve_points(work, *matrices, point_count, points, eigenvalues, &stage);
        } catch (const std::bad_alloc &) {
            stage = "host workspace allocation";
        }
        if (OpenMXGpuDeviceSynchronize() != OPENMX_GPU_SUCCESS) success = false;
        openmx_gemmul8TrimWorkspaces();
        (void)openmx_magma_release_z_workspace();
        (void)openmx_hipsolver_release_workspace();
    }
    openmx_mgpu_release_devices(admission);
    openmx_mgpu_scope_leave(previous);
    if (!success) {
        (void)OpenMXGpuGetLastError();
        if (OPENMX_STDOUT_LEVEL(2))
            std::fprintf(stderr, "<Band_DFT_kpath> GPU %d failed in %s\n", device, stage);
    }
    return success;
}

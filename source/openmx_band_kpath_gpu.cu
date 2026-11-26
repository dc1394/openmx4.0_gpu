#include "openmx_band_kpath_gpu.h"
#include "openmx_gpu_memory.hpp"
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_stdout_level.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <tuple>
#include <vector>

extern "C" {
OpenMXBlasStatus openmx_gemmul8Zgemm(OpenMXBlasHandle, OpenMXBlasOperation, OpenMXBlasOperation,
    int, int, int, const OpenMXGpuComplex *, const OpenMXGpuComplex *, int,
    const OpenMXGpuComplex *, int, const OpenMXGpuComplex *, OpenMXGpuComplex *, int);
void openmx_gemmul8TrimWorkspaces(void);
#if OPENMX_GPU_HIP
int openmx_magma_zheevdx_gpu(int, int, void *, double *, int *);
int openmx_magma_zheev_values_gpu(int, void *, double *);
int openmx_hipsolver_zheevd_gpu(int, int, void *, double *, int *);
int openmx_gpu_eigen_backend(int, int, int);
void openmx_magma_ozaki2_release(void);
int openmx_magma_release_z_workspace(void);
int openmx_hipsolver_release_workspace(void);
#endif
}

namespace {
using Z = OpenMXGpuComplex;
using openmx::gpu::DeviceBuffer;
constexpr int threads = 128;
constexpr double two_pi = 6.283185307179586476925;
constexpr size_t device_reserve = size_t{512} << 20;
constexpr int jacobi_max_order = 32;
constexpr int min_batched_points = 8;

struct BlockGroup {
    int first, exchange_begin, end;
};

struct LatticeTranslation {
    int x, y, z;
    bool operator<(const LatticeTranslation &other) const
    {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }
    bool operator==(const LatticeTranslation &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct DeviceBlock {
    int row, col, rows, cols, value_offset, translation;
};

__global__ void bloch_phases(int count, const LatticeTranslation *translations, OpenMXKPathPoint point,
                             const OpenMXKPathPoint *batch, Z *phases)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto t = translations[i];
        const auto k = batch ? batch[blockIdx.y] : point;
        const double kr = k.xyz[0] * t.x + k.xyz[1] * t.y + k.xyz[2] * t.z;
        double sine, cosine;
        sincos(two_pi * kr, &sine, &cosine);
        phases[(size_t)blockIdx.y * count + i] = OpenMXGpuMakeComplex(cosine, sine);
    }
}

__device__ void add_bloch(Z &sum, Z phase, double real, double imag)
{
    sum.x += phase.x * real - phase.y * imag;
    sum.y += phase.y * real + phase.x * imag;
}

/* Each group owns a dense atom-pair block. Sum periodic images in the
   original order, with one thread per orbital pair and no atomic adds. */
__global__ void assemble(int n, int matrix_order, int width, int spin, int spin_channels, bool lno,
                         bool noncollinear, bool imaginary_diagonals,
                         int translation_count, const BlockGroup *groups, const DeviceBlock *blocks,
                         const double *values, const Z *phases, Z *s, Z *h)
{
    phases += (size_t)blockIdx.y * translation_count;
    h += (size_t)blockIdx.y * matrix_order * matrix_order;
    if (s) s += (size_t)blockIdx.y * n * n;
    const BlockGroup group = groups[blockIdx.x];
    const DeviceBlock pair = blocks[group.first];
    for (int entry = threadIdx.x; entry < pair.rows * pair.cols; entry += blockDim.x) {
        Z overlap{}, up{}, down{}, mix{};
        for (int image = group.first; image < group.exchange_begin; ++image) {
            const DeviceBlock block = blocks[image];
            const double *v = values + block.value_offset + (size_t)entry * width;
            const Z phase = phases[block.translation];
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
        for (int image = group.exchange_begin; image < group.end; ++image) {
            const DeviceBlock block = blocks[image];
            const double *v = values + block.value_offset + (size_t)entry * spin_channels;
            add_bloch(up, phases[block.translation], v[spin], 0.0);
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
    vectors += (size_t)blockIdx.y * n * n;
    values += (size_t)blockIdx.y * n;
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
    values += (size_t)blockIdx.y * n;
    h += (size_t)blockIdx.y * order * order;
    discard += (size_t)blockIdx.y * n;
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
    discard += (size_t)blockIdx.y * n;
    h += (size_t)blockIdx.y * order * order;
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

#if OPENMX_GPU_HIP
#define OpenMXKPathZheevdBufferSize hipsolverDnZheevd_bufferSize
#define OpenMXKPathZheevd hipsolverDnZheevd
#define OpenMXKPathJacobiCreate hipsolverDnCreateSyevjInfo
#define OpenMXKPathJacobiDestroy hipsolverDnDestroySyevjInfo
#define OpenMXKPathJacobiTolerance hipsolverDnXsyevjSetTolerance
#define OpenMXKPathJacobiSweeps hipsolverDnXsyevjSetMaxSweeps
#define OpenMXKPathJacobiBufferSize hipsolverDnZheevjBatched_bufferSize
#define OpenMXKPathJacobiSolve hipsolverDnZheevjBatched
#define OpenMXKPathBeginCapture hipStreamBeginCapture
#define OpenMXKPathEndCapture hipStreamEndCapture
#define OpenMXKPathDestroyGraph hipGraphDestroy
#define OpenMXKPathDestroyExecutable hipGraphExecDestroy
#define OpenMXKPathLaunchGraph hipGraphLaunch
using JacobiInfo = hipsolverSyevjInfo_t;
using Graph = hipGraph_t;
using GraphExec = hipGraphExec_t;
constexpr auto capture_mode = hipStreamCaptureModeThreadLocal;
constexpr auto no_vectors = HIPSOLVER_EIG_MODE_NOVECTOR;
#else
#define OpenMXKPathZheevdBufferSize cusolverDnZheevd_bufferSize
#define OpenMXKPathZheevd cusolverDnZheevd
#define OpenMXKPathJacobiCreate cusolverDnCreateSyevjInfo
#define OpenMXKPathJacobiDestroy cusolverDnDestroySyevjInfo
#define OpenMXKPathJacobiTolerance cusolverDnXsyevjSetTolerance
#define OpenMXKPathJacobiSweeps cusolverDnXsyevjSetMaxSweeps
#define OpenMXKPathJacobiBufferSize cusolverDnZheevjBatched_bufferSize
#define OpenMXKPathJacobiSolve cusolverDnZheevjBatched
#define OpenMXKPathBeginCapture cudaStreamBeginCapture
#define OpenMXKPathEndCapture cudaStreamEndCapture
#define OpenMXKPathDestroyGraph cudaGraphDestroy
#define OpenMXKPathDestroyExecutable cudaGraphExecDestroy
#define OpenMXKPathLaunchGraph cudaGraphLaunch
using JacobiInfo = syevjInfo_t;
using Graph = cudaGraph_t;
using GraphExec = cudaGraphExec_t;
constexpr auto capture_mode = cudaStreamCaptureModeThreadLocal;
constexpr auto no_vectors = CUSOLVER_EIG_MODE_NOVECTOR;
#endif

OpenMXGpuError instantiate_graph(GraphExec *executable, Graph graph)
{
#if OPENMX_GPU_HIP
    return hipGraphInstantiate(executable, graph, nullptr, nullptr, 0);
#else
    return cudaGraphInstantiateWithFlags(executable, graph, 0);
#endif
}

/* Record on an isolated stream and submit on the stream already used by
   GEMMul8 and the copies. Only full batches reuse the fixed node arguments. */
struct BatchCapture {
    OpenMXGpuStream stream = nullptr;

    explicit BatchCapture(bool enabled)
    {
        if (enabled && OpenMXGpuStreamCreateWithFlags(&stream, OPENMX_GPU_STREAM_NONBLOCKING) != OPENMX_GPU_SUCCESS) {
            stream = nullptr;
            (void)OpenMXGpuGetLastError();
        }
    }
    ~BatchCapture() { if (stream) (void)OpenMXGpuStreamDestroy(stream); }
    BatchCapture(const BatchCapture &) = delete;
    BatchCapture &operator=(const BatchCapture &) = delete;
};

struct BatchGraph {
    BatchGraph() = default;
    ~BatchGraph() { if (executable) (void)OpenMXKPathDestroyExecutable(executable); }
    BatchGraph(const BatchGraph &) = delete;
    BatchGraph &operator=(const BatchGraph &) = delete;

    template<class Enqueue>
    bool submit(OpenMXGpuStream capture, OpenMXSolverHandle solver, const Enqueue &enqueue, int &replays)
    {
        if (!capture) return enqueue(nullptr);
        if (!attempted) {
            attempted = true;
            if (OpenMXSolverSetStream(solver, capture) == OPENMX_SOLVER_SUCCESS &&
                OpenMXKPathBeginCapture(capture, capture_mode) == OPENMX_GPU_SUCCESS) {
                const bool submitted = enqueue(capture);
                Graph graph = nullptr;
                const auto status = OpenMXKPathEndCapture(capture, &graph);
                if (graph) {
                    GraphExec ready = nullptr;
                    if (submitted && status == OPENMX_GPU_SUCCESS &&
                        instantiate_graph(&ready, graph) == OPENMX_GPU_SUCCESS)
                        executable = ready;
                    (void)OpenMXKPathDestroyGraph(graph);
                }
            }
            if (OpenMXSolverSetStream(solver, nullptr) != OPENMX_SOLVER_SUCCESS) return false;
            if (!executable) (void)OpenMXGpuGetLastError();
        }
        if (!executable) return enqueue(nullptr);
        if (OpenMXKPathLaunchGraph(executable, nullptr) != OPENMX_GPU_SUCCESS) return false;
        ++replays;
        return true;
    }

private:
    GraphExec executable = nullptr;
    bool attempted = false;
};

/* One handle and one scratch space for all k-points. Cache the queries
   separately for the overlap (vectors) and the transformed H (values). */
struct NativeEigensolver {
    OpenMXSolverHandle handle = nullptr;
    DeviceBuffer<Z> work;
    DeviceBuffer<double> eigenvalues;
    DeviceBuffer<int> info;
    int previous_order[2]{}, work_size[2]{};

    ~NativeEigensolver()
    {
        if (handle) (void)OpenMXSolverDestroy(handle);
    }

    bool solve(int order, Z *matrix, double *values, bool vectors)
    {
        const auto job = vectors ? OPENMX_SOLVER_EIG_VECTOR : no_vectors;
        if (!handle && OpenMXSolverCreate(&handle) != OPENMX_SOLVER_SUCCESS) return false;
        if (!info && !info.allocate(1)) return false;
        if (eigenvalues.size() < (size_t)order) {
            eigenvalues.reset();
            if (!eigenvalues.allocate(order)) return false;
        }
        int &required = work_size[vectors];
        if (previous_order[vectors] != order) {
            if (OpenMXKPathZheevdBufferSize(handle, job, OPENMX_BLAS_FILL_LOWER, order, matrix,
                                          order, eigenvalues.get(), &required) != OPENMX_SOLVER_SUCCESS ||
                required < 0) return false;
            if (work.size() < (size_t)required) {
                work.reset();
                if (!work.allocate(required)) return false;
            }
            previous_order[vectors] = order;
        }
        if (OpenMXKPathZheevd(handle, job, OPENMX_BLAS_FILL_LOWER, order, matrix, order,
                            eigenvalues.get(), work.get(), required, info.get()) != OPENMX_SOLVER_SUCCESS)
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
    DeviceBuffer<unsigned char> metadata;
    const DeviceBlock *blocks = nullptr;
    const BlockGroup *groups = nullptr;
    const LatticeTranslation *translations = nullptr;
    DeviceBuffer<double> values, overlap_values;
    DeviceBuffer<Z> phases, s, h, product;
    DeviceBuffer<unsigned char> discard;
    OpenMXBlasHandle blas = nullptr;
    NativeEigensolver solver;
    int group_count = 0, translation_count = 0;

    ~Workspace()
    {
        if (blas) (void)OpenMXBlasDestroy(blas);
    }

    bool prepare(const OpenMXKPathMatrices &m, int order)
    {
        const size_t n2 = (size_t)m.order * m.order, order2 = (size_t)order * order;
        if (order2 > std::numeric_limits<int>::max() / 4) return false;

        size_t free = 0, total = 0;
        const size_t dense = (n2 + 2 * order2) * sizeof(Z);
        const size_t solver_room = 2 * order2 * sizeof(Z);
        if (openmx_gpu_capacity_test_recovery() ||
            OpenMXGpuMemGetInfo(&free, &total) != OPENMX_GPU_SUCCESS ||
            free < dense + solver_room + device_reserve) return false;

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
            const auto exchange = std::lower_bound(ordered.begin() + first, ordered.begin() + i,
                m.exchange_offset, [](const auto &block, int offset) { return block.value_offset < offset; });
            spans.push_back({first, (int)(exchange - ordered.begin()), i});
        }
        group_count = (int)spans.size();

        std::vector<LatticeTranslation> shifts;
        shifts.reserve(ordered.size());
        for (const auto &block : ordered)
            shifts.push_back({block.translation[0], block.translation[1], block.translation[2]});
        std::sort(shifts.begin(), shifts.end());
        shifts.erase(std::unique(shifts.begin(), shifts.end()), shifts.end());
        std::vector<DeviceBlock> packed;
        packed.reserve(ordered.size());
        for (const auto &block : ordered) {
            const LatticeTranslation t{block.translation[0], block.translation[1], block.translation[2]};
            const int shift = (int)(std::lower_bound(shifts.begin(), shifts.end(), t) - shifts.begin());
            packed.push_back({block.row, block.col, block.rows, block.cols, block.value_offset, shift});
        }
        translation_count = (int)shifts.size();
        const size_t block_bytes = packed.size() * sizeof(DeviceBlock);
        const size_t group_bytes = spans.size() * sizeof(BlockGroup);
        const size_t translation_bytes = shifts.size() * sizeof(LatticeTranslation);
        const size_t metadata_bytes = block_bytes + group_bytes + translation_bytes;
        const size_t data = (size_t)m.value_count * sizeof(double) + metadata_bytes + shifts.size() * sizeof(Z);
        if (free - dense - solver_room - device_reserve < data) return false;

        if (!metadata.allocate(metadata_bytes) || !values.allocate(m.value_count) || !phases.allocate(shifts.size()) ||
            !s.allocate(n2) || !h.allocate(order2) || !product.allocate(order2) ||
            !overlap_values.allocate(m.order) || !discard.allocate(m.order) ||
            OpenMXBlasCreate(&blas) != OPENMX_BLAS_SUCCESS) return false;

        static_assert(alignof(DeviceBlock) == alignof(BlockGroup) && alignof(BlockGroup) == alignof(LatticeTranslation));
        std::vector<unsigned char> host_metadata(metadata_bytes);
        std::memcpy(host_metadata.data(), packed.data(), block_bytes);
        std::memcpy(host_metadata.data() + block_bytes, spans.data(), group_bytes);
        std::memcpy(host_metadata.data() + block_bytes + group_bytes, shifts.data(), translation_bytes);
        blocks = reinterpret_cast<const DeviceBlock *>(metadata.get());
        groups = reinterpret_cast<const BlockGroup *>(metadata.get() + block_bytes);
        translations = reinterpret_cast<const LatticeTranslation *>(metadata.get() + block_bytes + group_bytes);

        return OpenMXGpuMemcpy(metadata.get(), host_metadata.data(), metadata_bytes,
                               OPENMX_GPU_COPY_HOST_TO_DEVICE) == OPENMX_GPU_SUCCESS &&
               OpenMXGpuMemcpy(values.get(), m.values, values.size() * sizeof(double),
                               OPENMX_GPU_COPY_HOST_TO_DEVICE) == OPENMX_GPU_SUCCESS;
    }

    bool phase_rows(OpenMXKPathPoint point, int count, const OpenMXKPathPoint *batch, Z *output,
                    OpenMXGpuStream stream = nullptr) const
    {
        const dim3 grid(((size_t)translation_count + threads - 1) / threads, count);
        bloch_phases<<<grid, threads, 0, stream>>>(translation_count, translations, point, batch, output);
        return OpenMXGpuGetLastError() == OPENMX_GPU_SUCCESS;
    }

    bool construct(const OpenMXKPathMatrices &m, int order, int spin, int count,
                   const Z *phase, Z *overlap, Z *hamiltonian, OpenMXGpuStream stream = nullptr) const
    {
        if (OpenMXGpuMemsetAsync(hamiltonian, 0, (size_t)count * h.size() * sizeof(Z), stream) != OPENMX_GPU_SUCCESS ||
            (overlap && OpenMXGpuMemsetAsync(overlap, 0, (size_t)count * s.size() * sizeof(Z), stream) != OPENMX_GPU_SUCCESS))
            return false;
        const int width = m.noncollinear ? (m.imaginary_diagonals ? 7 : 5) :
                          (m.lno ? 2 * m.spin_channels : 1 + m.spin_channels);
        assemble<<<dim3(group_count, count), threads, 0, stream>>>(m.order, order, width, spin, m.spin_channels,
            m.lno, m.noncollinear, m.imaginary_diagonals, translation_count, groups,
            blocks, values.get(), phase, overlap, hamiltonian);
        return OpenMXGpuGetLastError() == OPENMX_GPU_SUCCESS;
    }

    bool diagonalize(int order, Z *matrix, double *eigen, bool vectors)
    {
#if OPENMX_GPU_HIP
        /* MAGMA's small driver can use LAPACK. The vector wrapper also
           handles rocSOLVER's known complex eigenvector failure cases. */
        if (OpenMXGpuDeviceSynchronize() != OPENMX_GPU_SUCCESS) return false;
        int found = order, status;
        if (order <= 128 || openmx_gpu_eigen_backend(order, order, 1)) {
            if (!vectors) return solver.solve(order, matrix, eigen, false) && good_spectrum(order, eigen);
            status = openmx_hipsolver_zheevd_gpu(order, order, matrix, eigen, &found);
        } else {
            status = vectors ? openmx_magma_zheevdx_gpu(order, order, matrix, eigen, &found)
                             : openmx_magma_zheev_values_gpu(order, matrix, eigen);
            // MAGMA destroys its solve streams. Release their Ozaki-II
            // scratch before the next solve creates new streams.
            openmx_magma_ozaki2_release();
        }
        return status == 0 && found == order && good_spectrum(order, eigen);
#else
        return solver.solve(order, matrix, eigen, vectors) && good_spectrum(order, eigen);
#endif
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

    bool transform(int n, int order, Z *hamiltonian, const Z *overlap, bool native_precision)
    {
        const auto zgemm = native_precision ? OpenMXBlasZgemm : openmx_gemmul8Zgemm;
        const Z one = OpenMXGpuMakeComplex(1.0, 0.0), zero = OpenMXGpuMakeComplex(0.0, 0.0);
        // U is block diagonal for the two spin components. Keep its n-by-n
        // storage and apply it to the two halves without zero products.
        for (int col = 0; col < order; col += n) {
            const size_t start = (size_t)col * order;
            if (zgemm(blas, OPENMX_BLAS_OP_N, OPENMX_BLAS_OP_N, order, n, n, &one,
                      hamiltonian + start, order, overlap, n, &zero, product.get() + start, order) !=
                OPENMX_BLAS_SUCCESS) return false;
        }
        for (int row = 0; row < order; row += n) {
            if (zgemm(blas, OPENMX_BLAS_OP_C, OPENMX_BLAS_OP_N, n, order, n, &one,
                      overlap, n, product.get() + row, order, &zero, hamiltonian + row, order) !=
                OPENMX_BLAS_SUCCESS) return false;
        }
        return true;
    }
};

bool penalize_overlap(int n, int order, double cutoff, int count, const double *values, Z *h,
                      unsigned char *discard, OpenMXGpuStream stream = nullptr)
{
    penalty_diagonal<<<dim3((n + threads - 1) / threads, count), threads, 0, stream>>>(
        n, order, cutoff, values, h, discard);
    if (OpenMXGpuGetLastError() != OPENMX_GPU_SUCCESS) return false;
    remove_penalized<<<dim3(((size_t)order * order + threads - 1) / threads, count), threads, 0, stream>>>(
        n, order, discard, h);
    return OpenMXGpuGetLastError() == OPENMX_GPU_SUCCESS;
}

/* Small contiguous Hermitian matrices share one FP64 Jacobi submission.
   Keep this handle and scratch separate from the per-point fallback. */
struct BatchedEigensolver {
    OpenMXSolverHandle handle = nullptr;
    JacobiInfo parameters = nullptr;
    DeviceBuffer<Z> work;
    DeviceBuffer<int> info;
    std::vector<int> statuses;

    ~BatchedEigensolver()
    {
        if (parameters) (void)OpenMXKPathJacobiDestroy(parameters);
        if (handle) (void)OpenMXSolverDestroy(handle);
    }

    bool prepare(int n, int order, int capacity, const Z *s, const Z *h,
                 const double *overlap, const double *eigen, size_t remaining_bytes)
    {
        if (OpenMXSolverCreate(&handle) != OPENMX_SOLVER_SUCCESS ||
            OpenMXKPathJacobiCreate(&parameters) != OPENMX_SOLVER_SUCCESS ||
            OpenMXKPathJacobiTolerance(parameters, 1.0e-14) != OPENMX_SOLVER_SUCCESS ||
            OpenMXKPathJacobiSweeps(parameters, 100) != OPENMX_SOLVER_SUCCESS)
            return false;

        int overlap_work = 0, eigen_work = 0;
        if (OpenMXKPathJacobiBufferSize(handle, OPENMX_SOLVER_EIG_VECTOR, OPENMX_BLAS_FILL_LOWER,
                n, s, n, overlap, &overlap_work, parameters, capacity) != OPENMX_SOLVER_SUCCESS ||
            OpenMXKPathJacobiBufferSize(handle, no_vectors, OPENMX_BLAS_FILL_LOWER,
                order, h, order, eigen, &eigen_work, parameters, capacity) != OPENMX_SOLVER_SUCCESS ||
            overlap_work < 0 || eigen_work < 0)
            return false;
        const size_t required = (size_t)std::max({1, overlap_work, eigen_work});
        statuses.resize(capacity);
        return required <= remaining_bytes / sizeof(Z) && work.allocate(required) && info.allocate(capacity);
    }

    bool enqueue(int order, int count, Z *matrix, double *spectrum, bool vectors, OpenMXGpuStream capture)
    {
        // Detect a captured solver that leaves a status from the previous batch.
        if (capture && OpenMXGpuMemsetAsync(info.get(), 0xff, (size_t)count * sizeof(int), capture) != OPENMX_GPU_SUCCESS)
            return false;
        const auto job = vectors ? OPENMX_SOLVER_EIG_VECTOR : no_vectors;
        return OpenMXKPathJacobiSolve(handle, job, OPENMX_BLAS_FILL_LOWER, order, matrix, order,
            spectrum, work.get(), (int)work.size(), info.get(), parameters, count) == OPENMX_SOLVER_SUCCESS;
    }

    bool read(int order, int count, const double *spectrum, double *values)
    {
        return OpenMXGpuMemcpy(statuses.data(), info.get(), (size_t)count * sizeof(int),
                               OPENMX_GPU_COPY_DEVICE_TO_HOST) == OPENMX_GPU_SUCCESS &&
               std::all_of(statuses.begin(), statuses.begin() + count, [](int i) { return i == 0; }) &&
               OpenMXGpuMemcpy(values, spectrum, (size_t)count * order * sizeof(double),
                               OPENMX_GPU_COPY_DEVICE_TO_HOST) == OPENMX_GPU_SUCCESS;
    }
};

bool solve_batched(Workspace &w, const OpenMXKPathMatrices &m, int order, int point_count,
                   const OpenMXKPathPoint *points, double *eigen)
{
    constexpr int max_batch = 64;
    constexpr size_t batch_limit = size_t{32} << 20;
    constexpr double min_overlap_ratio = 1.0e-5;
    const int n = m.order;
    const size_t per_point = (w.phases.size() + w.s.size() + w.h.size()) * sizeof(Z) +
        (size_t)(n + order) * sizeof(double) + (size_t)n + sizeof(OpenMXKPathPoint) + sizeof(int);
    size_t free = 0, total = 0;
    if (OpenMXGpuMemGetInfo(&free, &total) != OPENMX_GPU_SUCCESS || free <= device_reserve)
        return false;
    const size_t budget = std::min(batch_limit, free - device_reserve);
    const int capacity = (int)std::min({(size_t)max_batch, (size_t)point_count, budget / (2 * per_point)});
    if (capacity < min_batched_points) return false;

    DeviceBuffer<OpenMXKPathPoint> device_points;
    DeviceBuffer<Z> phases, s, h;
    DeviceBuffer<double> overlap_values, eigenvalues;
    DeviceBuffer<unsigned char> discard;
    BatchedEigensolver solver;
    std::vector<double> values((size_t)capacity * order);
    if (!device_points.allocate(capacity) || !phases.allocate((size_t)capacity * w.phases.size()) ||
        !s.allocate((size_t)capacity * w.s.size()) || !h.allocate((size_t)capacity * w.h.size()) ||
        !overlap_values.allocate((size_t)capacity * n) || !eigenvalues.allocate((size_t)capacity * order) ||
        !discard.allocate((size_t)capacity * n) ||
        !solver.prepare(n, order, capacity, s.get(), h.get(), overlap_values.get(), eigenvalues.get(),
                        budget - (size_t)capacity * per_point))
        return false;

    // Amortize capture over several full batches on the same GPU owner.
    BatchCapture capture(point_count / capacity >= 4);
    std::vector<BatchGraph> assembly_graphs(m.spin_channels);
    BatchGraph eigen_graph;
    // On failure, drain the stream before freeing the graphs and device storage.
    struct PendingBatch {
        bool finished = false;
        ~PendingBatch() { if (!finished) (void)OpenMXGpuStreamSynchronize(nullptr); }
    } pending;
    int replays = 0;
    for (int first = 0; first < point_count;) {
        const int count = std::min(capacity, point_count - first);
        // Warm up each solver mode before the first capture.
        OpenMXGpuStream capture_stream = first > 0 && count == capacity ? capture.stream : nullptr;
        if (OpenMXGpuMemcpy(device_points.get(), points + first, (size_t)count * sizeof(*points),
                            OPENMX_GPU_COPY_HOST_TO_DEVICE) != OPENMX_GPU_SUCCESS)
            return false;
        for (int spin = 0; spin < m.spin_channels; ++spin) {
            const bool overlap_needed = spin == 0 || m.lno;
            const auto assemble_batch = [&](OpenMXGpuStream stream) {
                return (spin != 0 || w.phase_rows({}, count, device_points.get(), phases.get(), stream)) &&
                    w.construct(m, order, spin, count, phases.get(), overlap_needed ? s.get() : nullptr, h.get(), stream) &&
                    (!overlap_needed || solver.enqueue(n, count, s.get(), overlap_values.get(), true, stream));
            };
            if (!assembly_graphs[spin].submit(overlap_needed ? capture_stream : nullptr, solver.handle,
                                               assemble_batch, replays))
                return false;
            if (overlap_needed) {
                if (!solver.read(n, count, overlap_values.get(), values.data())) return false;
                /* The standard solver applies the overlap penalties. Also use
                   it when scaling could amplify the Jacobi solver's error. */
                for (int k = 0; k < count; ++k) {
                    const double *row = values.data() + (size_t)k * n;
                    if (!good_spectrum(n, row)) return false;
                    const double limit = std::max({0.0, m.overlap_cutoff, row[n - 1] * min_overlap_ratio});
                    if (row[0] <= limit) return false;
                }
                scale_overlap<<<dim3((w.s.size() + threads - 1) / threads, count), threads>>>(
                    n, s.get(), overlap_values.get());
                if (OpenMXGpuGetLastError() != OPENMX_GPU_SUCCESS) return false;
            }

            for (int k = 0; k < count; ++k)
                if (!w.transform(n, order, h.get() + (size_t)k * w.h.size(),
                                 s.get() + (size_t)k * w.s.size(), false)) return false;
            const auto solve_hamiltonian = [&](OpenMXGpuStream stream) {
                return (m.lno || penalize_overlap(n, order, m.overlap_cutoff, count,
                            overlap_values.get(), h.get(), discard.get(), stream)) &&
                       solver.enqueue(order, count, h.get(), eigenvalues.get(), false, stream);
            };
            if (!eigen_graph.submit(capture_stream, solver.handle, solve_hamiltonian, replays) ||
                !solver.read(order, count, eigenvalues.get(), values.data())) return false;
            for (int k = 0; k < count; ++k) {
                const double *row = values.data() + (size_t)k * order;
                if (!good_spectrum(order, row)) return false;
                std::copy_n(row, order, eigen + ((size_t)(first + k) * m.spin_channels + spin) * order);
            }
        }
        first += count;
    }
    pending.finished = true;
    if (OPENMX_STDOUT_LEVEL(2)) {
        std::printf("<Band_DFT_kpath> FP64 batched eigensolver, up to %d k-points per batch\n", capacity);
        if (replays > 0)
            std::printf("<Band_DFT_kpath> %s Graph submissions: %d\n", OPENMX_GPU_RUNTIME_NAME, replays);
    }
    return true;
}

bool solve_points(Workspace &w, const OpenMXKPathMatrices &m, int point_count,
                  const OpenMXKPathPoint *points, double *eigen, const char **stage)
{
    const int order = m.noncollinear ? 2 * m.order : m.order;
    const int n = m.order;
    *stage = "device workspace allocation";
    if (!w.prepare(m, order)) return false;

    if (order <= jacobi_max_order && point_count >= min_batched_points) {
        *stage = "batched eigensolve";
        try {
            if (solve_batched(w, m, order, point_count, points, eigen)) return true;
        } catch (const std::bad_alloc &) {
            // Optional batch buffers are gone before the per-point retry.
        }
        if (OpenMXGpuStreamSynchronize(nullptr) != OPENMX_GPU_SUCCESS) return false;
        (void)OpenMXGpuGetLastError();
    }

    std::vector<double> overlap(n);
    for (int k = 0; k < point_count; ++k) {
        *stage = "Bloch phases";
        if (!w.phase_rows(points[k], 1, nullptr, w.phases.get())) return false;

        for (int spin = 0; spin < m.spin_channels; ++spin) {
            const bool overlap_needed = spin == 0 || m.lno;
            *stage = "Bloch matrix assembly";
            if (!w.construct(m, order, spin, 1, w.phases.get(), overlap_needed ? w.s.get() : nullptr, w.h.get())) return false;
            *stage = "overlap eigensolve and scaling";
            if (overlap_needed && !w.orthogonalize(n, overlap.data())) return false;

            *stage = "Hamiltonian transformation";
            const bool ill_conditioned = *std::min_element(overlap.begin(), overlap.end()) < m.overlap_cutoff;
            if (!w.transform(n, order, w.h.get(), w.s.get(), ill_conditioned)) return false;
            if (!m.lno && !penalize_overlap(n, order, m.overlap_cutoff, 1, w.overlap_values.get(),
                                           w.h.get(), w.discard.get())) return false;

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

extern "C" int openmx_band_kpath_solve_gpu(int device, const OpenMXKPathMatrices *matrices,
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
#if OPENMX_GPU_HIP
        openmx_magma_ozaki2_release();
#endif
        Workspace work;
        try {
            success = solve_points(work, *matrices, point_count, points, eigenvalues, &stage);
        } catch (const std::bad_alloc &) {
            stage = "host workspace allocation";
        }
        if (OpenMXGpuDeviceSynchronize() != OPENMX_GPU_SUCCESS) success = false;
        openmx_gemmul8TrimWorkspaces();
#if OPENMX_GPU_HIP
        (void)openmx_magma_release_z_workspace();
        (void)openmx_hipsolver_release_workspace();
#endif
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

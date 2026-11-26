#include <openmx/core/radial.hpp>
#include <openmx/core/bessel.hpp>
#include <openmx_core_host_simd.h>
#include "host_radial_kernels.hpp"
#include "radial_weight.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace openmx::core {
namespace {

void validate_grid(const RadialGrid& grid)
{
  if (grid.lmax < 0 || grid.lmax > 29 || grid.intervals < 1 ||
      grid.intervals == INT_MAX || grid.k_count < 0 ||
      !std::isfinite(grid.rmin) || grid.rmin < 0 ||
      !std::isfinite(grid.radial_step) || grid.radial_step <= 0 ||
      !std::isfinite(grid.k_step) || grid.k_step < 0)
    throw std::invalid_argument("invalid radial transform grid");
  const double rmax = grid.rmin + grid.intervals*grid.radial_step;
  const double xmax = rmax*std::max(0, grid.k_count-1)*grid.k_step;
  if (!std::isfinite(rmax) || !std::isfinite(xmax) ||
      xmax > (INT_MAX-grid.lmax-11)/1.5)
    throw std::invalid_argument("radial transform exceeds Bessel recurrence range");
}

void validate_quadrature(const RadialQuadrature& grid)
{
  if ((grid.weighting != RadialWeighting::trapezoid && grid.weighting != RadialWeighting::gauss) ||
      grid.lmax < 0 || grid.lmax > 29 ||
      grid.radii.size() < (grid.weighting == RadialWeighting::gauss ? 1u : 2u) ||
      grid.radii.size() > INT_MAX || grid.wavevectors.size() > INT_MAX ||
      grid.measures.size() != grid.radii.size() ||
      !std::isfinite(grid.step) || grid.step <= 0)
    throw std::invalid_argument("invalid radial quadrature shape");
  double max_k = 0;
  for (double k : grid.wavevectors) {
    if (!std::isfinite(k) || k < 0) throw std::invalid_argument("invalid radial wavevector");
    max_k = std::max(k,max_k);
  }
  for (std::size_t i = 0; i < grid.radii.size(); ++i) {
    if (!std::isfinite(grid.radii[i]) || grid.radii[i] < 0 ||
        (i && grid.radii[i] <= grid.radii[i-1]) ||
        !std::isfinite(grid.measures[i]) || grid.measures[i] < 0)
      throw std::invalid_argument("invalid radial quadrature coordinate or measure");
  }
  const double max_x = grid.radii.back()*max_k;
  if (!std::isfinite(max_x) || max_x > (INT_MAX-grid.lmax-11)/1.5)
    throw std::invalid_argument("radial transform exceeds Bessel recurrence range");
}

void validate_samples(const RadialSamples& samples)
{
  validate_quadrature(samples.grid);
  if (samples.orbitals.size() > INT_MAX ||
      samples.values.extent(0) != samples.orbitals.size() ||
      samples.values.extent(1) != samples.grid.radii.size())
    throw std::invalid_argument("radial sample shape does not match grid");
  for (const auto& orbital : samples.orbitals)
    if (orbital.angular < 0 || orbital.angular > samples.grid.lmax || orbital.multiplicity < 0)
      throw std::invalid_argument("invalid radial orbital");
  if (samples.products.size() > INT_MAX)
    throw std::invalid_argument("too many radial products");
  for (const auto& product : samples.products)
    if (product.left < 0 || product.right < 0 ||
        static_cast<std::size_t>(product.left) >= samples.orbitals.size() ||
        static_cast<std::size_t>(product.right) >= samples.orbitals.size() ||
        product.angular < 0 || product.angular > samples.grid.lmax)
      throw std::invalid_argument("invalid radial product indices");
  for (std::size_t i = 0; i < samples.values.size(); ++i)
    if (!std::isfinite(samples.values.data()[i]))
      throw std::invalid_argument("non-finite radial sample");
}

ProductChannel channel_at(const RadialSamples& samples, std::size_t row)
{
  return samples.products.empty() ? ProductChannel{static_cast<int>(row),-1,samples.orbitals[row].angular}
                                  : samples.products[row];
}

template<class ExecutionSpace>
class CompleteBeforeRelease {
 public:
  explicit CompleteBeforeRelease(const ExecutionSpace& execution) : execution_(execution) {}
  ~CompleteBeforeRelease() { if (pending_) execution_.fence("OpenMX radial scratch lifetime"); }
  CompleteBeforeRelease(const CompleteBeforeRelease&) = delete;
  CompleteBeforeRelease& operator=(const CompleteBeforeRelease&) = delete;
  void begin() noexcept { pending_ = true; }
  void finish() {
    execution_.fence("OpenMX radial transform before host commit");
    pending_ = false;
  }
 private:
  const ExecutionSpace& execution_;
  bool pending_ = false;
};

template<class ExecutionSpace>
Tensor<double, 2> transform_kokkos(const RadialSamples& samples)
{
  constexpr bool host_execution =
    Kokkos::SpaceAccessibility<ExecutionSpace, Kokkos::HostSpace>::accessible;
  using Matrix = Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace>;
  using BesselLayout = std::conditional_t<host_execution, Kokkos::LayoutRight, Kokkos::LayoutLeft>;
  using Cube = Kokkos::View<double***, BesselLayout, ExecutionSpace>;
  using Indices = Kokkos::View<int**, Kokkos::LayoutRight, ExecutionSpace>;
  using Vector = Kokkos::View<double*, ExecutionSpace>;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using HostInput = Kokkos::View<const double**, Kokkos::LayoutRight, Kokkos::HostSpace, Unmanaged>;
  using HostVector = Kokkos::View<const double*, Kokkos::HostSpace, Unmanaged>;
  using HostOutput = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace, Unmanaged>;
  using Policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;
  const auto& grid = samples.grid;
  const std::size_t orbital_count = samples.orbitals.size();
  const std::size_t channel_count = samples.products.empty() ? orbital_count : samples.products.size();
  const int radial_count = static_cast<int>(grid.radii.size());
  const int k_count = static_cast<int>(grid.wavevectors.size());
  const int lmax = grid.lmax;
  const double step = grid.step;
  const auto weighting = grid.weighting;
  Tensor<double, 2> result({channel_count, static_cast<std::size_t>(k_count)});
  if (channel_count == 0 || k_count == 0) return result;

  const ExecutionSpace execution;
  Matrix radial(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.samples"), orbital_count, radial_count);
  const HostInput host_radial(samples.values.data(), orbital_count, radial_count);
  Indices channels(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.channels"), channel_count, 3);
  auto host_channels = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, channels);
  for (std::size_t row = 0; row < channel_count; ++row) {
    const auto product = channel_at(samples,row);
    host_channels(row,0) = product.angular;
    host_channels(row,1) = product.left;
    host_channels(row,2) = product.right;
  }
  Vector radii(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.radii"), radial_count);
  Vector measures(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.measures"), radial_count);
  Vector wavevectors(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.wavevectors"), k_count);
  const HostVector host_radii(grid.radii.data(), radial_count);
  const HostVector host_measures(grid.measures.data(), radial_count);
  const HostVector host_wavevectors(grid.wavevectors.data(), k_count);

  // Amortize launches and host synchronization without making scratch grow
  // with the full reciprocal grid. Even a single wavevector may exceed the
  // budget for an unusually large input; it is still the minimum work unit.
  constexpr std::size_t scratch_budget = 8u*1024u*1024u;
  const std::size_t bytes_per_k = sizeof(double)*
    (static_cast<std::size_t>(lmax+1)*radial_count + channel_count);
  std::size_t capacity = host_execution ? 32 : std::max(std::size_t{1}, scratch_budget/bytes_per_k);
  // Align device rows to 256 bytes when the memory budget permits it.
  if (!host_execution && capacity >= 32) capacity -= capacity%32;
  const int batch_size = static_cast<int>(std::min(static_cast<std::size_t>(k_count), capacity));
  const int batch_capacity = host_execution ? batch_size : static_cast<int>(std::min(capacity,
    ((static_cast<std::size_t>(batch_size)+31)/32)*32));
  // Keep a bounded result on the device and copy it directly into the
  // caller's result once. Larger results reuse the original tile buffer.
  constexpr std::size_t output_budget = 2u*1024u*1024u;
  const bool copy_all = !host_execution && result.size() <= output_budget/sizeof(double);
  // Device lanes vary over k and read adjacent Bessel values. Host execution
  // retains contiguous radial rows. Both preserve the ascending radial sum.
  Cube bessel(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.bessel"), batch_capacity, lmax+1, radial_count);
  Matrix output(Kokkos::view_alloc(Kokkos::WithoutInitializing, "openmx.radial.output"),
    channel_count, copy_all ? k_count : batch_capacity);
  decltype(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, output)) host_output;
  if (!copy_all) host_output = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, output);
  // Declared after the Views: even exceptional exits complete outstanding
  // work before their storage is released.
  CompleteBeforeRelease<ExecutionSpace> completion(execution);
  completion.begin();
  Kokkos::deep_copy(execution, radial, host_radial);
  Kokkos::deep_copy(execution, channels, host_channels);
  Kokkos::deep_copy(execution, radii, host_radii);
  Kokkos::deep_copy(execution, measures, host_measures);
  Kokkos::deep_copy(execution, wavevectors, host_wavevectors);
  auto destination = result.view();
  for (int first = 0; first < k_count;) {
    const int count = std::min(batch_size, k_count-first);
    const int output_first = copy_all ? first : 0;
    completion.begin();
    Kokkos::parallel_for("openmx.radial.bessel", Policy(execution, 0, static_cast<std::size_t>(count)*radial_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const int k = static_cast<int>(host_execution ? index/radial_count : index%count);
        const int i = static_cast<int>(host_execution ? index%radial_count : index/count);
        const double x = wavevectors(first+k)*radii(i);
        double sb[32];
        spherical_bessel_values(x, lmax, sb);
        for (int l = 0; l <= lmax; ++l)
          bessel(k,l,i) = radial_weight(sb[l], measures(i), i, radial_count-1, weighting);
      });
    Kokkos::parallel_for("openmx.radial.integrate", Policy(execution, 0, static_cast<std::size_t>(count)*channel_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const std::size_t orbital = index/count;
        const int k = static_cast<int>(index%count);
        const int angular = channels(orbital,0);
        const int left = channels(orbital,1), right = channels(orbital,2);
        double sum = 0.0;
        for (int i = 0; i < radial_count; ++i) {
          double term = bessel(k,angular,i)*radial(left,i);
          if (right >= 0) term *= radial(right,i);
          sum += term;
        }
        output(orbital,output_first+k) = sum*step;
      });
    if (!copy_all) {
      Kokkos::deep_copy(execution, host_output, output);
      completion.finish();
      for (std::size_t orbital = 0; orbital < channel_count; ++orbital)
        for (int k = 0; k < count; ++k)
          element(destination, orbital, first+k) = host_output(orbital,k);
    }
    first += count;
  }
  if (copy_all) {
    const HostOutput host_result(result.data(),channel_count,k_count);
    Kokkos::deep_copy(execution,host_result,output);
    completion.finish();
  }
  return result;
}

} // namespace

RadialQuadrature uniform_quadrature(RadialGrid grid)
{
  validate_grid(grid);
  RadialQuadrature result{grid.lmax, {}, {}, {}, grid.radial_step};
  result.radii.resize(grid.intervals+1);
  result.measures.resize(grid.intervals+1);
  result.wavevectors.resize(grid.k_count);
  for (int i = 0; i <= grid.intervals; ++i) {
    const double r = grid.rmin+i*grid.radial_step;
    result.radii[i] = r;
    result.measures[i] = r*r;
  }
  for (int k = 0; k < grid.k_count; ++k) result.wavevectors[k] = k*grid.k_step;
  validate_quadrature(result);
  return result;
}

RadialQuadrature squared_radius_quadrature(int intervals, double rmin, double rmax,
    double kmin, double kmax, std::span<const double> gauss_nodes)
{
  if (intervals < 1 || intervals == INT_MAX || !std::isfinite(rmin) || rmin < 0 ||
      !std::isfinite(rmax) || rmax <= rmin || !std::isfinite(kmin) || kmin < 0 ||
      !std::isfinite(kmax) || kmax < kmin || gauss_nodes.size() > INT_MAX)
    throw std::invalid_argument("invalid squared-radius quadrature bounds");
  const double xmin = std::sqrt(rmin), xmax = std::sqrt(rmax);
  const double step = (xmax-xmin)/intervals;
  RadialQuadrature result{0, {}, {}, {}, step};
  result.radii.resize(intervals+1);
  result.measures.resize(intervals+1);
  result.wavevectors.reserve(gauss_nodes.size());
  for (int i = 0; i <= intervals; ++i) {
    const double x = xmin+i*step;
    const double r = x*x;
    result.radii[i] = r;
    result.measures[i] = 2.0*r*r*x;
  }
  for (double node : gauss_nodes) {
    if (!std::isfinite(node) || node < -1 || node > 1)
      throw std::invalid_argument("invalid Gauss-Legendre node");
    result.wavevectors.push_back(0.50*((kmax-kmin)*node+(kmax+kmin)));
  }
  validate_quadrature(result);
  return result;
}

RadialQuadrature gauss_quadrature(int lmax, double rmin, double rmax,
    double kmin, double kmax, std::span<const double> radial_nodes,
    std::span<const double> radial_weights, std::span<const double> wavevector_nodes)
{
  if (lmax < 0 || lmax > 29 || radial_nodes.empty() || radial_nodes.size() > INT_MAX ||
      radial_weights.size() != radial_nodes.size() || wavevector_nodes.size() > INT_MAX ||
      !std::isfinite(rmin) || rmin < 0 || !std::isfinite(rmax) || rmax <= rmin ||
      !std::isfinite(kmin) || kmin < 0 || !std::isfinite(kmax) || kmax < kmin)
    throw std::invalid_argument("invalid Gauss radial quadrature");
  const double dr = rmax-rmin, sr = rmax+rmin, dk = kmax-kmin, sk = kmax+kmin;
  RadialQuadrature result{lmax,{},{},{},0.5*dr,RadialWeighting::gauss};
  for (std::size_t i = 0; i < radial_nodes.size(); ++i) {
    const double node = radial_nodes[i], weight = radial_weights[i];
    if (!std::isfinite(node) || node < -1 || node > 1 ||
        !std::isfinite(weight) || weight <= 0)
      throw std::invalid_argument("invalid Gauss radial node or weight");
    const double r = 0.5*(dr*node+sr);
    result.radii.push_back(r);
    result.measures.push_back(r*r*weight);
  }
  for (double node : wavevector_nodes) {
    if (!std::isfinite(node) || node < -1 || node > 1)
      throw std::invalid_argument("invalid Gauss wavevector node");
    result.wavevectors.push_back(0.5*(dk*node+sk));
  }
  validate_quadrature(result);
  return result;
}

RadialSamples sample_radial(RadialGrid grid, std::span<const int> multiplicities,
                           const Sampler& sample)
{
  validate_grid(grid);
  if (multiplicities.size() != static_cast<std::size_t>(grid.lmax+1) || !sample)
    throw std::invalid_argument("missing radial orbital description or sampler");
  std::vector<Orbital> orbitals;
  for (int l = 0; l <= grid.lmax; ++l) {
    if (multiplicities[l] < 0 || static_cast<std::size_t>(multiplicities[l]) > INT_MAX-orbitals.size())
      throw std::invalid_argument("invalid radial multiplicity");
    for (int m = 0; m < multiplicities[l]; ++m) orbitals.push_back({l,m});
  }
  Tensor<double,2> values({orbitals.size(), static_cast<std::size_t>(grid.intervals+1)});
  auto view = values.view();
  for (std::size_t orbital = 0; orbital < orbitals.size(); ++orbital)
    for (int i = 0; i <= grid.intervals; ++i) {
      const double value = sample(orbitals[orbital].angular, orbitals[orbital].multiplicity,
                                  grid.rmin + i*grid.radial_step);
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite radial sample");
      element(view, orbital, i) = value;
    }
  return {uniform_quadrature(grid), std::move(orbitals), std::move(values), {}};
}

namespace {

Tensor<double,2> transform_host_samples(const RadialSamples& samples, int mode)
{
  const auto& grid = samples.grid;
  const int radial_count = static_cast<int>(grid.radii.size());
  const int k_count = static_cast<int>(grid.wavevectors.size());
  const auto channel_count = samples.products.empty() ? samples.orbitals.size() : samples.products.size();
  Tensor<double,2> result({channel_count,static_cast<std::size_t>(k_count)});
  if (channel_count == 0 || k_count == 0) return result;
  if (grid.lmax == 0 && channel_count == 1 && samples.products.empty()) {
    // The neutral potential has one output row. Consume each weight in
    // place, without building a table that only one contraction reads.
    for (int k = 0; k < k_count; ++k) {
      double sum = 0.0;
      for (int i = 0; i < radial_count; ++i) {
        double sb;
        spherical_bessel_values(grid.wavevectors[k]*grid.radii[i],0,&sb);
        sum += radial_weight(sb,grid.measures[i],i,radial_count-1,grid.weighting)*samples.values.data()[i];
      }
      result.data()[k] = sum*grid.step;
    }
    return result;
  }
  constexpr std::size_t scratch_budget = 8u*1024u*1024u;
  // With few channels per Bessel order, forming the SIMD weight tiles
  // costs more than the contractions save. Keep those narrow cases scalar.
  const std::size_t orders = grid.lmax+1;
  const auto* kernels = host::openmx_host_radial_kernels(mode);
  if (channel_count >= 6*orders && kernels && kernels->width > 1 && k_count >= kernels->width &&
      grid.radii.size() <= scratch_budget/(sizeof(double)*orders*kernels->width)) {
    std::vector<ProductChannel> channels;
    if (samples.products.empty()) {
      channels.reserve(channel_count);
      for (std::size_t row = 0; row < channel_count; ++row) channels.push_back(channel_at(samples,row));
    }
    kernels->transform({grid.lmax,radial_count,k_count,channel_count,grid.weighting,grid.step,
      grid.radii.data(),grid.measures.data(),grid.wavevectors.data(),samples.values.data(),
      samples.products.empty() ? channels.data() : samples.products.data(),result.data()});
    return result;
  }
  Tensor<double,2> bessel({static_cast<std::size_t>(grid.lmax+1),grid.radii.size()});
  auto work = bessel.view();
  const auto input = samples.values.view();
  auto output = result.view();
  for (int k = 0; k < k_count; ++k) {
    for (int i = 0; i < radial_count; ++i) {
      double sb[32];
      spherical_bessel_values(grid.wavevectors[k]*grid.radii[i],grid.lmax,sb);
      for (int l = 0; l <= grid.lmax; ++l)
        element(work,l,i) = radial_weight(sb[l],grid.measures[i],i,radial_count-1,grid.weighting);
    }
    for (std::size_t orbital = 0; orbital < channel_count; ++orbital) {
      const auto product = channel_at(samples,orbital);
      double sum = 0.0;
      for (int i = 0; i < radial_count; ++i) {
        double term = element(work,product.angular,i)*element(input,product.left,i);
        if (product.right >= 0) term *= element(input,product.right,i);
        sum += term;
      }
      element(output,orbital,k) = sum*grid.step;
    }
  }
  return result;
}
} // namespace

Tensor<double,2> transform_host(const RadialSamples& samples)
{
  validate_samples(samples);
  return transform_host_samples(samples,OPENMX_GRID_SCALAR);
}

bool has_accelerator() noexcept
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  return true;
#else
  return false;
#endif
}

bool Context::start(int device)
{
  if (finished_) throw std::logic_error("OpenMX core context already finished");
  if (runtime_) {
    if (device != device_) { fallback(HostFallback::device_changed); return false; }
    return true;
  }
  // An externally owned runtime must outlive this context. Never finalize it.
  if (Kokkos::is_initialized()) { fallback(HostFallback::external_runtime); return false; }
  if (Kokkos::is_finalized())
    throw std::logic_error("Kokkos cannot be reinitialized; retain the application context");
  auto settings = Kokkos::InitializationSettings().set_disable_warnings(true);
  if (device >= 0) settings.set_device_id(device);
  runtime_ = std::make_unique<Kokkos::ScopeGuard>(settings);
  device_ = device;
#if defined(KOKKOS_ENABLE_SYCL)
  const auto selected = Kokkos::DefaultExecutionSpace().sycl_queue().get_device();
  capabilities_ = {OPENMX_ACCELERATOR_SYCL,
      selected.has(sycl::aspect::fp64) ? OPENMX_FP64_NATIVE : OPENMX_FP64_UNAVAILABLE,
      selected.get_info<sycl::info::device::host_unified_memory>() ?
          OPENMX_MEMORY_SHARED : OPENMX_MEMORY_DISCRETE, 0};
#elif defined(KOKKOS_ENABLE_CUDA)
  capabilities_ = {OPENMX_ACCELERATOR_CUDA, OPENMX_FP64_NATIVE, OPENMX_MEMORY_UNKNOWN, 0};
#elif defined(KOKKOS_ENABLE_HIP)
  capabilities_ = {OPENMX_ACCELERATOR_HIP, OPENMX_FP64_NATIVE, OPENMX_MEMORY_UNKNOWN, 0};
#endif
  return true;
}

void Context::fallback(HostFallback reason)
{
  last_fallback_ = reason;
  const unsigned bit = 1u << static_cast<unsigned>(reason);
  const char* verbose = std::getenv("OPENMX_CORE_VERBOSE");
  if ((reported_fallbacks_ & bit) || !verbose || std::atoi(verbose) == 0) return;
  const char* message = reason == HostFallback::external_runtime ? "Kokkos runtime owned by caller" :
      reason == HostFallback::device_changed ? "requested device differs from context device" :
      reason == HostFallback::device_memory ? "device workspace allocation failed" :
      "selected device does not support native FP64";
  std::fprintf(stderr, "<OpenMX_Core> radial transform uses CPU: %s\n", message);
  reported_fallbacks_ |= bit;
}

Tensor<double,2> Context::transform(const RadialSamples& samples, int device)
{
  validate_samples(samples);
  if (finished_) throw std::logic_error("OpenMX core context already finished");
  last_fallback_ = HostFallback::none;
  const auto on_host = [&] {
    return transform_host_samples(samples,host_simd_ ? *host_simd_ : openmx_core_host_simd_mode(nullptr));
  };
  // GPU-enabled Kokkos initializes its GPU even for a Serial dispatch. A CPU
  // request must remain usable on machines without a visible accelerator.
  if (host_only_ || (has_accelerator() && device < 0)) return on_host();
  if (!has_accelerator()) device = -1;
  // A later input can change legacy rank/device mapping. Until execution-space
  // instances are migrated, keep that calculation on the host safely.
  if (!start(device)) return on_host();
  if (!openmx_accelerator_can_run_fp64(capabilities_)) {
    fallback(HostFallback::fp64_unavailable);
    return on_host();
  }
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  if constexpr (std::is_same_v<ExecutionSpace,Kokkos::Serial>) return on_host();
  try {
    return transform_kokkos<ExecutionSpace>(samples);
  } catch (const Kokkos::Experimental::BadAlloc& error) {
    if (!has_accelerator() || error.memory_space_name() != ExecutionSpace::memory_space::name()) throw;
    fallback(HostFallback::device_memory);
    return on_host();
  }
}

} // namespace openmx::core

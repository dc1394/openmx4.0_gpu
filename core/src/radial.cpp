#include <openmx/core/radial.hpp>
#include <openmx/core/bessel.hpp>

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

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

void validate_samples(const RadialSamples& samples)
{
  validate_grid(samples.grid);
  if (samples.orbitals.size() > INT_MAX ||
      samples.values.extent(0) != samples.orbitals.size() ||
      samples.values.extent(1) != static_cast<std::size_t>(samples.grid.intervals+1))
    throw std::invalid_argument("radial sample shape does not match grid");
  for (const auto& orbital : samples.orbitals)
    if (orbital.angular < 0 || orbital.angular > samples.grid.lmax || orbital.multiplicity < 0)
      throw std::invalid_argument("invalid radial orbital");
}

// Both execution paths use this weighting and integrate in ascending radial
// order. Parallelism is over independent wavevectors/orbitals, not the sum.
KOKKOS_INLINE_FUNCTION
double radial_weight(double bessel, double radius, int point, int intervals)
{
  double weighted = bessel*(radius*radius);
  if (point == 0 || point == intervals) weighted *= 0.5;
  return weighted;
}

template<class ExecutionSpace>
class CompleteBeforeRelease {
 public:
  explicit CompleteBeforeRelease(const ExecutionSpace& execution) : execution_(execution) {}
  ~CompleteBeforeRelease() { execution_.fence("OpenMX radial scratch lifetime"); }
  CompleteBeforeRelease(const CompleteBeforeRelease&) = delete;
  CompleteBeforeRelease& operator=(const CompleteBeforeRelease&) = delete;
 private:
  const ExecutionSpace& execution_;
};

template<class ExecutionSpace>
Tensor<double, 2> transform_kokkos(const RadialSamples& samples)
{
  using Matrix = Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace>;
  using Cube = Kokkos::View<double***, Kokkos::LayoutRight, ExecutionSpace>;
  using Indices = Kokkos::View<int*, ExecutionSpace>;
  using Policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;
  const auto grid = samples.grid;
  const std::size_t orbital_count = samples.orbitals.size();
  const std::size_t radial_count = grid.intervals+1;
  Tensor<double, 2> result({orbital_count, static_cast<std::size_t>(grid.k_count)});
  if (orbital_count == 0 || grid.k_count == 0) return result;

  const ExecutionSpace execution;
  Matrix radial("openmx.radial.samples", orbital_count, radial_count);
  auto host_radial = Kokkos::create_mirror_view(radial);
  const auto input = samples.values.view();
  for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
    for (std::size_t i = 0; i < radial_count; ++i)
      host_radial(orbital, i) = element(input, orbital, i);
  Indices angular("openmx.radial.angular", orbital_count);
  auto host_angular = Kokkos::create_mirror_view(angular);
  for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
    host_angular(orbital) = samples.orbitals[orbital].angular;
  // Bound scratch memory independently of the full reciprocal-space grid.
  const int batch_size = std::min(32, grid.k_count);
  Cube bessel("openmx.radial.bessel", batch_size, grid.lmax+1, radial_count);
  Matrix output("openmx.radial.output", orbital_count, batch_size);
  auto host_output = Kokkos::create_mirror_view(output);
  // Declared after the Views: even exceptional exits complete outstanding
  // work before their storage is released.
  CompleteBeforeRelease<ExecutionSpace> completion(execution);
  Kokkos::deep_copy(execution, radial, host_radial);
  Kokkos::deep_copy(execution, angular, host_angular);
  auto destination = result.view();
  for (int first = 0; first < grid.k_count;) {
    const int count = std::min(batch_size, grid.k_count-first);
    Kokkos::parallel_for("openmx.radial.bessel", Policy(execution, 0, count*radial_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const int k = static_cast<int>(index/radial_count);
        const int i = static_cast<int>(index%radial_count);
        const double radius = grid.rmin + i*grid.radial_step;
        double sb[32];
        spherical_bessel(((first+k)*grid.k_step)*radius, grid.lmax, sb, nullptr);
        for (int l = 0; l <= grid.lmax; ++l)
          bessel(k,l,i) = radial_weight(sb[l], radius, i, grid.intervals);
      });
    Kokkos::parallel_for("openmx.radial.integrate", Policy(execution, 0, count*orbital_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const std::size_t orbital = index/count;
        const int k = static_cast<int>(index%count);
        double sum = 0.0;
        for (int i = 0; i <= grid.intervals; ++i)
          sum += radial(orbital,i)*bessel(k,angular(orbital),i);
        output(orbital,k) = sum*grid.radial_step;
      });
    Kokkos::deep_copy(execution, host_output, output);
    execution.fence("OpenMX radial transform before host commit");
    for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
      for (int k = 0; k < count; ++k)
        element(destination, orbital, first+k) = host_output(orbital,k);
    first += count;
  }
  return result;
}

} // namespace

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
  return {grid, std::move(orbitals), std::move(values)};
}

Tensor<double,2> transform_host(const RadialSamples& samples)
{
  validate_samples(samples);
  const auto& grid = samples.grid;
  Tensor<double,2> result({samples.orbitals.size(), static_cast<std::size_t>(grid.k_count)});
  Tensor<double,2> bessel({static_cast<std::size_t>(grid.lmax+1), static_cast<std::size_t>(grid.intervals+1)});
  auto work = bessel.view();
  const auto input = samples.values.view();
  auto output = result.view();
  for (int k = 0; k < grid.k_count; ++k) {
    const double wavevector = k*grid.k_step;
    for (int i = 0; i <= grid.intervals; ++i) {
      const double radius = grid.rmin + i*grid.radial_step;
      double sb[32];
      spherical_bessel(wavevector*radius, grid.lmax, sb, nullptr);
      for (int l = 0; l <= grid.lmax; ++l)
        element(work, l, i) = radial_weight(sb[l], radius, i, grid.intervals);
    }
    for (std::size_t orbital = 0; orbital < samples.orbitals.size(); ++orbital) {
      double sum = 0.0;
      for (int i = 0; i <= grid.intervals; ++i)
        sum += element(input, orbital, i)*element(work, samples.orbitals[orbital].angular, i);
      element(output, orbital, k) = sum*grid.radial_step;
    }
  }
  return result;
}

bool has_accelerator() noexcept
{
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return true;
#else
  return false;
#endif
}

bool Context::start(int device)
{
  if (finished_) throw std::logic_error("OpenMX core context already finished");
  if (runtime_) return device == device_;
  // An externally owned runtime must outlive this context. Never finalize it.
  if (Kokkos::is_initialized()) return false;
  if (Kokkos::is_finalized())
    throw std::logic_error("Kokkos cannot be reinitialized; retain the application context");
  auto settings = Kokkos::InitializationSettings().set_disable_warnings(true);
  if (device >= 0) settings.set_device_id(device);
  runtime_ = std::make_unique<Kokkos::ScopeGuard>(settings);
  device_ = device;
  return true;
}

Tensor<double,2> Context::transform(const RadialSamples& samples, int device)
{
  validate_samples(samples);
  if (finished_) throw std::logic_error("OpenMX core context already finished");
  // GPU-enabled Kokkos initializes its GPU even for a Serial dispatch. A CPU
  // request must remain usable on machines without a visible accelerator.
  if (has_accelerator() && device < 0) return transform_host(samples);
  if (!has_accelerator()) device = -1;
  // A later input can change legacy rank/device mapping. Until execution-space
  // instances are migrated, keep that calculation on the host safely.
  if (!start(device)) return transform_host(samples);
  return transform_kokkos<Kokkos::DefaultExecutionSpace>(samples);
}

} // namespace openmx::core

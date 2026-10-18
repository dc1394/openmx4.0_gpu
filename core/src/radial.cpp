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

void validate_quadrature(const RadialQuadrature& grid)
{
  if (grid.lmax < 0 || grid.lmax > 29 || grid.radii.size() < 2 ||
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
  for (std::size_t i = 0; i < samples.values.size(); ++i)
    if (!std::isfinite(samples.values.data()[i]))
      throw std::invalid_argument("non-finite radial sample");
}

// Both execution paths use this weighting and integrate in ascending radial
// order. Parallelism is over independent wavevectors/orbitals, not the sum.
KOKKOS_INLINE_FUNCTION
double radial_weight(double bessel, double measure, int point, int intervals)
{
  double weighted = bessel*measure;
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
  using Vector = Kokkos::View<double*, ExecutionSpace>;
  using Policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>;
  const auto& grid = samples.grid;
  const std::size_t orbital_count = samples.orbitals.size();
  const int radial_count = static_cast<int>(grid.radii.size());
  const int k_count = static_cast<int>(grid.wavevectors.size());
  const int lmax = grid.lmax;
  const double step = grid.step;
  Tensor<double, 2> result({orbital_count, static_cast<std::size_t>(k_count)});
  if (orbital_count == 0 || k_count == 0) return result;

  const ExecutionSpace execution;
  Matrix radial("openmx.radial.samples", orbital_count, radial_count);
  auto host_radial = Kokkos::create_mirror_view(radial);
  const auto input = samples.values.view();
  for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
    for (int i = 0; i < radial_count; ++i)
      host_radial(orbital, i) = element(input, orbital, i);
  Indices angular("openmx.radial.angular", orbital_count);
  auto host_angular = Kokkos::create_mirror_view(angular);
  for (std::size_t orbital = 0; orbital < orbital_count; ++orbital)
    host_angular(orbital) = samples.orbitals[orbital].angular;
  Vector radii("openmx.radial.radii", radial_count);
  Vector measures("openmx.radial.measures", radial_count);
  Vector wavevectors("openmx.radial.wavevectors", k_count);
  auto host_radii = Kokkos::create_mirror_view(radii);
  auto host_measures = Kokkos::create_mirror_view(measures);
  auto host_wavevectors = Kokkos::create_mirror_view(wavevectors);
  for (int i = 0; i < radial_count; ++i) {
    host_radii(i) = grid.radii[i];
    host_measures(i) = grid.measures[i];
  }
  for (int k = 0; k < k_count; ++k) host_wavevectors(k) = grid.wavevectors[k];
  // Bound scratch memory independently of the full reciprocal-space grid.
  const int batch_size = std::min(32, k_count);
  Cube bessel("openmx.radial.bessel", batch_size, lmax+1, radial_count);
  Matrix output("openmx.radial.output", orbital_count, batch_size);
  auto host_output = Kokkos::create_mirror_view(output);
  // Declared after the Views: even exceptional exits complete outstanding
  // work before their storage is released.
  CompleteBeforeRelease<ExecutionSpace> completion(execution);
  Kokkos::deep_copy(execution, radial, host_radial);
  Kokkos::deep_copy(execution, angular, host_angular);
  Kokkos::deep_copy(execution, radii, host_radii);
  Kokkos::deep_copy(execution, measures, host_measures);
  Kokkos::deep_copy(execution, wavevectors, host_wavevectors);
  auto destination = result.view();
  for (int first = 0; first < k_count;) {
    const int count = std::min(batch_size, k_count-first);
    Kokkos::parallel_for("openmx.radial.bessel", Policy(execution, 0, static_cast<std::size_t>(count)*radial_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const int k = static_cast<int>(index/radial_count);
        const int i = static_cast<int>(index%radial_count);
        const double radius = radii(i);
        double sb[32];
        spherical_bessel(wavevectors(first+k)*radius, lmax, sb, nullptr);
        for (int l = 0; l <= lmax; ++l)
          bessel(k,l,i) = radial_weight(sb[l], measures(i), i, radial_count-1);
      });
    Kokkos::parallel_for("openmx.radial.integrate", Policy(execution, 0, count*orbital_count),
      KOKKOS_LAMBDA(std::size_t index) {
        const std::size_t orbital = index/count;
        const int k = static_cast<int>(index%count);
        double sum = 0.0;
        for (int i = 0; i < radial_count; ++i)
          sum += radial(orbital,i)*bessel(k,angular(orbital),i);
        output(orbital,k) = sum*step;
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
  return {uniform_quadrature(grid), std::move(orbitals), std::move(values)};
}

Tensor<double,2> transform_host(const RadialSamples& samples)
{
  validate_samples(samples);
  const auto& grid = samples.grid;
  const int radial_count = static_cast<int>(grid.radii.size());
  const int k_count = static_cast<int>(grid.wavevectors.size());
  Tensor<double,2> result({samples.orbitals.size(),static_cast<std::size_t>(k_count)});
  if (samples.orbitals.empty() || k_count == 0) return result;
  Tensor<double,2> bessel({static_cast<std::size_t>(grid.lmax+1),grid.radii.size()});
  auto work = bessel.view();
  const auto input = samples.values.view();
  auto output = result.view();
  for (int k = 0; k < k_count; ++k) {
    for (int i = 0; i < radial_count; ++i) {
      double sb[32];
      spherical_bessel(grid.wavevectors[k]*grid.radii[i],grid.lmax,sb,nullptr);
      for (int l = 0; l <= grid.lmax; ++l)
        element(work,l,i) = radial_weight(sb[l],grid.measures[i],i,radial_count-1);
    }
    for (std::size_t orbital = 0; orbital < samples.orbitals.size(); ++orbital) {
      double sum = 0.0;
      for (int i = 0; i < radial_count; ++i)
        sum += element(input,orbital,i)*element(work,samples.orbitals[orbital].angular,i);
      element(output,orbital,k) = sum*grid.step;
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

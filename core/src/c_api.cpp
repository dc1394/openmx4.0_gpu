#include "openmx_core.h"
#include "openmx_core_radial.h"
#include "pao_compat.hpp"
#include "vps_compat.hpp"
#include <openmx/core/radial.hpp>
#include <openmx/core/radial_table.hpp>
#include <openmx/core/bessel.hpp>
#include <openmx/core/fft_grid.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <exception>
#include <algorithm>

struct OpenMXCoreContext {
  openmx::core::Context implementation;
  std::unique_ptr<openmx::core::compat::PaoOwner> pao;
  std::unique_ptr<openmx::core::compat::VpsOwner> vps;
};

namespace {

template<class Function>
int checked_call(char* error, std::size_t capacity, Function&& function) noexcept
{
  if (error && capacity) error[0] = '\0';
  try {
    function();
    return 1;
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error,capacity,"%s",exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unexpected OpenMX core exception");
  }
  return 0;
}

std::span<const double> mesh(const double* radii, int count)
{
  if (!radii || count < 3) throw std::invalid_argument("missing radial table mesh");
  return {radii,static_cast<std::size_t>(count)};
}

openmx::core::RadialChannel channel(int angular, const double* values, int count)
{
  if (!values) throw std::invalid_argument("missing radial table row");
  return {angular,{values,static_cast<std::size_t>(count)}};
}

openmx::core::RadialQuadrature uniform_grid(int lmax, const OpenMXUniformRadialGrid* grid)
{
  if (!grid) throw std::invalid_argument("missing uniform radial grid");
  return openmx::core::uniform_quadrature({lmax,grid->intervals,grid->k_count,
      grid->rmin,grid->radial_step,grid->k_step});
}

void transform_table(OpenMXCoreContext* context, int device,
    const openmx::core::RadialTableView& table, openmx::core::RadialQuadrature grid,
    std::span<double* const> outputs)
{
  if (outputs.size() != table.channel_count())
    throw std::invalid_argument("radial output channel count mismatch");
  for (double* row : outputs)
    if (!row && !grid.wavevectors.empty()) throw std::invalid_argument("missing radial output row");
  auto samples = openmx::core::sample_table(table,std::move(grid));
  auto result = context ? context->implementation.transform(samples,device)
                        : openmx::core::transform_host(samples);
  const auto count = result.extent(1);
  if (count == 0) return;
  // No callbacks or allocations after this point: failure cannot leave a
  // partially updated species, and input/output aliasing is safe.
  for (std::size_t row = 0; row < outputs.size(); ++row)
    std::copy_n(result.data()+row*count,count,outputs[row]);
}

} // namespace

extern "C" int openmx_core_pao_allocate(OpenMXCoreContext* context, const OpenMXPaoShape* shape,
    OpenMXPaoViews* views, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!context || !shape || !views || context->pao || context->implementation.finished())
      throw std::invalid_argument("PAO allocation requires an empty context, shape and output views");
    if (shape->species <= 0 || shape->angular_channels <= 0 || shape->multiplicity <= 0 ||
        shape->radial_points <= 0 || shape->reciprocal_points <= 0)
      throw std::invalid_argument("PAO storage dimensions must be positive");
    auto owner = std::make_unique<openmx::core::compat::PaoOwner>(openmx::core::PaoShape{
        static_cast<std::size_t>(shape->species),static_cast<std::size_t>(shape->angular_channels),
        static_cast<std::size_t>(shape->multiplicity),static_cast<std::size_t>(shape->radial_points),
        static_cast<std::size_t>(shape->reciprocal_points)});
    // Both numeric allocations and compatibility pointer tables must succeed
    // before publishing any address. Nothing below this line can throw.
    *views = owner->views();
    context->pao = std::move(owner);
  });
}

extern "C" void openmx_core_pao_release(OpenMXCoreContext* context)
{
  if (context) context->pao.reset();
}

extern "C" int openmx_core_vps_allocate(OpenMXCoreContext* context, const OpenMXVpsShape* shape,
    const OpenMXNeutralProjectorShape* neutral_shape, OpenMXVpsViews* views,
    char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!context || !shape || !views || context->vps || context->implementation.finished())
      throw std::invalid_argument("VPS allocation requires an unfinished context without VPS storage");
    if (shape->species <= 0 || shape->spins < 1 || shape->spins > 2 || shape->projectors <= 0 ||
        shape->radial_points <= 0 || shape->reciprocal_points <= 0)
      throw std::invalid_argument("invalid VPS storage dimensions");
    const openmx::core::VpsShape dimensions{static_cast<std::size_t>(shape->species),
      static_cast<std::size_t>(shape->spins),static_cast<std::size_t>(shape->projectors),
      static_cast<std::size_t>(shape->radial_points),static_cast<std::size_t>(shape->reciprocal_points)};
    std::optional<openmx::core::NeutralProjectorShape> neutral;
    if (neutral_shape) {
      if (neutral_shape->angular_channels <= 0 || neutral_shape->multiplicity <= 0 || neutral_shape->wavevectors <= 0)
        throw std::invalid_argument("invalid neutral-projector storage dimensions");
      neutral = openmx::core::NeutralProjectorShape{dimensions.species,
        static_cast<std::size_t>(neutral_shape->angular_channels),
        static_cast<std::size_t>(neutral_shape->multiplicity),dimensions.radial_points,
        static_cast<std::size_t>(neutral_shape->wavevectors)};
    }
    auto owner = std::make_unique<openmx::core::compat::VpsOwner>(dimensions,neutral);
    *views = owner->views();
    context->vps = std::move(owner);
  });
}

extern "C" void openmx_core_vps_release(OpenMXCoreContext* context)
{
  if (context) context->vps.reset();
}

extern "C" int openmx_core_reciprocal_sample(OpenMXReciprocalKind kind, int count,
    double step, const double* values, double wavevector, double* result)
{
  return checked_call(nullptr,0,[&] {
    if (!result || !values || count <= 0 ||
        (kind != OPENMX_RECIPROCAL_PAO && kind != OPENMX_RECIPROCAL_PROJECTOR))
      throw std::invalid_argument("invalid reciprocal interpolation arguments");
    const double value = openmx::core::sample_reciprocal_row(
        kind == OPENMX_RECIPROCAL_PAO ? openmx::core::ReciprocalKind::pao
                                     : openmx::core::ReciprocalKind::projector,
        {values,static_cast<std::size_t>(count)},step,wavevector);
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite reciprocal interpolation result");
    *result = value;
  });
}

extern "C" int openmx_core_radial_sample(OpenMXRadialKind kind, int angular,
    int count, const double* radii, const double* values,
    double cutoff, double radius, double* result)
{
  return checked_call(nullptr,0,[&] {
    if (!result || !values) throw std::invalid_argument("missing radial interpolation row or output");
    openmx::core::RadialKind policy;
    switch (kind) {
      case OPENMX_RADIAL_PAO: policy = openmx::core::RadialKind::pao; break;
      case OPENMX_RADIAL_PROJECTOR: policy = openmx::core::RadialKind::projector; break;
      case OPENMX_RADIAL_NEUTRAL_POTENTIAL: policy = openmx::core::RadialKind::neutral_potential; break;
      default: throw std::invalid_argument("unknown radial interpolation kind");
    }
    const auto coordinates = mesh(radii,count);
    const double value = openmx::core::sample_radial_row(policy,angular,coordinates,
        {values,coordinates.size()},cutoff,radius);
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite radial interpolation result");
    *result = value;
  });
}

extern "C" int openmx_core_transform_pao(OpenMXCoreContext* context, int device,
    const OpenMXPaoTable* input, const OpenMXUniformRadialGrid* grid,
    char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input || input->lmax < 0 || input->lmax > 29 || !input->multiplicities)
      throw std::invalid_argument("invalid PAO table description");
    const auto radii = mesh(input->radii,input->mesh_count);
    auto quadrature = uniform_grid(input->lmax,grid);
    std::vector<openmx::core::RadialChannel> channels;
    std::vector<double*> outputs;
    for (int l = 0; l <= input->lmax; ++l) {
      const int count = input->multiplicities[l];
      if (count < 0 || static_cast<std::size_t>(count) > INT_MAX-channels.size())
        throw std::invalid_argument("invalid PAO table multiplicity");
      if (count && (!input->values || !input->values[l] || !input->output || !input->output[l]))
        throw std::invalid_argument("missing PAO table angular block");
      for (int m = 0; m < count; ++m) {
        channels.push_back(channel(l,input->values[l][m],input->mesh_count));
        outputs.push_back(input->output[l][m]);
      }
    }
    const openmx::core::RadialTableView table(openmx::core::RadialKind::pao,radii,channels);
    transform_table(context,device,table,std::move(quadrature),outputs);
  });
}

extern "C" int openmx_core_transform_projectors(OpenMXCoreContext* context, int device,
    const OpenMXProjectorTable* input, const OpenMXUniformRadialGrid* grid,
    char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input || input->spin_count < 1 || input->spin_count > 2 ||
        input->projector_count < 0 || input->projector_count > INT_MAX/input->spin_count ||
        (input->projector_count && !input->angular))
      throw std::invalid_argument("invalid nonlocal-projector table description");
    const auto radii = mesh(input->radii,input->mesh_count);
    std::vector<openmx::core::RadialChannel> channels;
    std::vector<double*> outputs;
    int lmax = 0;
    for (int spin = 0; spin < input->spin_count; ++spin) {
      if (input->projector_count && (!input->values[spin] || !input->output[spin]))
        throw std::invalid_argument("missing nonlocal-projector spin block");
      for (int p = 0; p < input->projector_count; ++p) {
        channels.push_back(channel(input->angular[p],input->values[spin][p],input->mesh_count));
        outputs.push_back(input->output[spin][p]);
        lmax = std::max(lmax,input->angular[p]);
      }
    }
    auto quadrature = uniform_grid(lmax,grid);
    const openmx::core::RadialTableView table(openmx::core::RadialKind::projector,radii,channels);
    transform_table(context,device,table,std::move(quadrature),outputs);
  });
}

extern "C" int openmx_core_transform_neutral_potential(OpenMXCoreContext* context, int device,
    const OpenMXNeutralTable* input, const OpenMXSquaredRadialGrid* grid,
    char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input || !grid || grid->k_count < 0 || (grid->k_count && !grid->gauss_nodes))
      throw std::invalid_argument("invalid neutral-potential table description");
    const auto radii = mesh(input->radii,input->mesh_count);
    const auto row = channel(0,input->values,input->mesh_count);
    const openmx::core::RadialTableView table(openmx::core::RadialKind::neutral_potential,
                                         radii,{&row,1},input->cutoff);
    auto quadrature = openmx::core::squared_radius_quadrature(grid->intervals,
        grid->rmin,grid->rmax,grid->kmin,grid->kmax,
        {grid->gauss_nodes,static_cast<std::size_t>(grid->k_count)});
    transform_table(context,device,table,std::move(quadrature),{&input->output,1});
  });
}

extern "C" const char* openmx_core_version(void) { return "4.0 GPU"; }
extern "C" int openmx_core_fft_grid_supported(int size)
{
  return openmx::core::fft_grid_supported(size);
}

extern "C" int openmx_core_run(int argc, char** argv, OpenMXCoreEntry entry)
{
  if (!entry) return 1;
  try {
    OpenMXCoreContext context;
    return entry(argc, argv, &context);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "OpenMX core: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "OpenMX core: unexpected exception\n");
  }
  return 1;
}

extern "C" void openmx_core_finish(OpenMXCoreContext* context)
{
  if (context) {
    context->pao.reset();
    context->vps.reset();
    context->implementation.finish();
  }
}

extern "C" int openmx_core_has_accelerator(void)
{
  return openmx::core::has_accelerator();
}

extern "C" int openmx_core_spherical_bessel(double x, int lmax, double* sb, double* derivative)
{
  if (!sb || !derivative || !std::isfinite(x) || x < 0 || lmax < 0 || lmax > 29 ||
      x > (INT_MAX-lmax-11)/1.5) return 0;
  openmx::core::spherical_bessel(x,lmax,sb,derivative);
  return 1;
}

extern "C" int openmx_core_pao_transform(OpenMXCoreContext* context, int device,
    int lmax, const int* multiplicities, int radial_intervals, int k_count,
    double rmin, double radial_step, double k_step, void* user,
    OpenMXRadialSample sample, OpenMXRadialStore store,
    char* error, std::size_t error_capacity)
{
  if (error && error_capacity) error[0] = '\0';
  try {
    if (lmax < 0 || lmax > 29 || !multiplicities || !sample || !store)
      throw std::invalid_argument("invalid PAO transform interface arguments");
    auto samples = openmx::core::sample_radial(
      {lmax, radial_intervals, k_count, rmin, radial_step, k_step},
      {multiplicities, static_cast<std::size_t>(lmax+1)},
      [=](int l, int m, double radius) { return sample(user, l, m, radius); });
    auto result = context ? context->implementation.transform(samples, device)
                          : openmx::core::transform_host(samples);
    const auto values = result.view();
    for (std::size_t orbital = 0; orbital < samples.orbitals.size(); ++orbital)
      for (int k = 0; k < k_count; ++k)
        store(user, samples.orbitals[orbital].angular, samples.orbitals[orbital].multiplicity,
              k, openmx::core::element(values, orbital, k));
    return 1;
  } catch (const std::exception& exception) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "%s", exception.what());
  } catch (...) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "unexpected PAO transform exception");
  }
  return 0;
}

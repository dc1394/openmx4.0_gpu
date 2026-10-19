#include "openmx_core.h"
#include "openmx_core_radial.h"
#include "openmx_core_neutral.h"
#include "openmx_core_pair_bessel.h"
#include "openmx_core_overlap.h"
#include "openmx_core_nonlocal.h"
#include "pao_compat.hpp"
#include "vps_compat.hpp"
#include "product_compat.hpp"
#include <openmx/core/radial.hpp>
#include <openmx/core/radial_table.hpp>
#include <openmx/core/neutral_projectors.hpp>
#include <openmx/core/bessel.hpp>
#include <openmx/core/fft_grid.hpp>
#include <openmx/core/pair_bessel.hpp>
#include <openmx/core/overlap_workspace.hpp>
#include <openmx/core/nonlocal_workspace.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <exception>
#include <algorithm>

struct OpenMXCoreContext {
  openmx::core::Context implementation;
  std::unique_ptr<openmx::core::compat::PaoOwner> pao;
  std::unique_ptr<openmx::core::compat::VpsOwner> vps;
  std::unique_ptr<openmx::core::compat::ProductOwner> products;
};

struct OpenMXPairBesselWorkspace {
  openmx::core::PairBesselWorkspace implementation;
};

struct OpenMXOverlapWorkspace {
  explicit OpenMXOverlapWorkspace(openmx::core::OverlapShape shape) : implementation(shape) {}
  openmx::core::OverlapWorkspace implementation;
};

struct OpenMXNonlocalWorkspace {
  OpenMXNonlocalWorkspace(openmx::core::NonlocalShape shape,
      std::span<const openmx::core::NonlocalSpecies> species,
      std::span<const double> wavevectors, std::size_t limit)
    : implementation(shape,species,wavevectors,limit) {}
  openmx::core::NonlocalWorkspace implementation;
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

openmx::core::RadialQuadrature gauss_grid(int lmax, const OpenMXGaussRadialGrid* grid)
{
  if (!grid || grid->radial_count < 1 || grid->k_count < 0 ||
      !grid->radial_nodes || !grid->radial_weights || (grid->k_count && !grid->wavevector_nodes))
    throw std::invalid_argument("invalid Gauss radial grid");
  return openmx::core::gauss_quadrature(lmax,grid->rmin,grid->rmax,grid->kmin,grid->kmax,
      {grid->radial_nodes,static_cast<std::size_t>(grid->radial_count)},
      {grid->radial_weights,static_cast<std::size_t>(grid->radial_count)},
      {grid->wavevector_nodes,static_cast<std::size_t>(grid->k_count)});
}

void rectangular_basis(int lmax, int multiplicity)
{
  if (lmax < 0 || lmax > 29 || multiplicity <= 0 || multiplicity > INT_MAX/(lmax+1))
    throw std::invalid_argument("invalid rectangular radial basis");
}

void transform_samples(OpenMXCoreContext* context, int device,
    const openmx::core::RadialSamples& samples,
    std::span<double* const> outputs)
{
  const auto channels = samples.products.empty() ? samples.orbitals.size() : samples.products.size();
  if (outputs.size() != channels)
    throw std::invalid_argument("radial output channel count mismatch");
  for (double* row : outputs)
    if (!row && !samples.grid.wavevectors.empty()) throw std::invalid_argument("missing radial output row");
  auto result = context ? context->implementation.transform(samples,device)
                        : openmx::core::transform_host(samples);
  const auto count = result.extent(1);
  if (count == 0) return;
  // No callbacks or allocations after this point: failure cannot leave a
  // partially updated species, and input/output aliasing is safe.
  for (std::size_t row = 0; row < outputs.size(); ++row)
    std::copy_n(result.data()+row*count,count,outputs[row]);
}

void transform_table(OpenMXCoreContext* context, int device,
    const openmx::core::RadialTableView& table, openmx::core::RadialQuadrature grid,
    std::span<double* const> outputs)
{
  transform_samples(context,device,openmx::core::sample_table(table,std::move(grid)),outputs);
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

extern "C" int openmx_core_product_allocate(OpenMXCoreContext* context, const OpenMXProductShape* shape,
    OpenMXProductViews* views, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!context || !shape || !views || context->products || context->implementation.finished())
      throw std::invalid_argument("product allocation requires an unfinished context without product storage");
    if (shape->species <= 0 || shape->wavevectors <= 0 || !shape->lmax || !shape->multiplicities)
      throw std::invalid_argument("invalid product storage shape");
    std::vector<openmx::core::ProductSpecies> species;
    for (int s = 0; s < shape->species; ++s) {
      if (shape->lmax[s] < 0 || shape->lmax[s] > 14 || !shape->multiplicities[s])
        throw std::invalid_argument("invalid product species basis");
      species.push_back({{shape->multiplicities[s],static_cast<std::size_t>(shape->lmax[s]+1)}});
    }
    auto owner = std::make_unique<openmx::core::compat::ProductOwner>(species,shape->wavevectors);
    *views = owner->views();
    context->products = std::move(owner);
  });
}

extern "C" void openmx_core_product_release(OpenMXCoreContext* context)
{
  if (context) context->products.reset();
}

extern "C" int openmx_core_transform_products(OpenMXCoreContext* context, int device,
    const OpenMXProductTable* input, const OpenMXGaussRadialGrid* grid, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input || input->lmax < 0 || input->lmax > 14 || !input->multiplicities)
      throw std::invalid_argument("invalid orbital-product table");
    const openmx::core::ProductLayout layout({input->multiplicities,static_cast<std::size_t>(input->lmax+1)});
    const auto radii = mesh(input->radii,input->mesh_count);
    auto quadrature = gauss_grid(2*input->lmax,grid);
    std::vector<openmx::core::RadialChannel> channels;
    std::vector<double*> outputs;
    for (int l1 = 0; l1 <= input->lmax; ++l1) {
      if (input->multiplicities[l1] && (!input->values || !input->values[l1] ||
          !input->output || !input->output[l1]))
        throw std::invalid_argument("missing product angular group");
      for (int m1 = 0; m1 < input->multiplicities[l1]; ++m1) {
        channels.push_back(channel(l1,input->values[l1][m1],input->mesh_count));
        if (!input->output[l1][m1]) throw std::invalid_argument("missing product left orbital");
        for (int l2 = l1; l2 <= input->lmax; ++l2) {
          if (input->multiplicities[l2] && !input->output[l1][m1][l2])
            throw std::invalid_argument("missing product right angular group");
          for (int m2 = 0; m2 < input->multiplicities[l2]; ++m2) {
            const auto block = input->output[l1][m1][l2][m2];
            if (!block) throw std::invalid_argument("missing product block");
            for (int l = 0; l <= 2*l2; ++l) {
              if (!block[l] && grid->k_count) throw std::invalid_argument("missing product output row");
              outputs.push_back(block[l]);
            }
          }
        }
      }
    }
    const openmx::core::RadialTableView table(openmx::core::RadialKind::pao,radii,channels);
    auto samples = openmx::core::sample_product_table(table,layout,std::move(quadrature));
    transform_samples(context,device,samples,outputs);
  });
}

extern "C" OpenMXPairBesselWorkspace* openmx_core_pair_bessel_create(int maximum_angular,
    int points, const double* wavevectors, std::size_t cache_limit, char* error, std::size_t capacity)
{
  OpenMXPairBesselWorkspace* result = nullptr;
  checked_call(error,capacity,[&] {
    if (!wavevectors || points <= 0) throw std::invalid_argument("missing pair Bessel wavevectors");
    result = new OpenMXPairBesselWorkspace{{maximum_angular,
        {wavevectors,static_cast<std::size_t>(points)},cache_limit}};
  });
  return result;
}

extern "C" int openmx_core_pair_bessel_sample(OpenMXPairBesselWorkspace* workspace,
    double distance, int angular, OpenMXPairBesselTable* table, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!workspace || !table) throw std::invalid_argument("missing pair Bessel workspace or output");
    const auto result = workspace->implementation.sample(distance,angular);
    const auto points = result.values.extent(2);
    *table = {result.values.data_handle(),
              &openmx::core::element(result.values,1,0,0),points,
              static_cast<std::size_t>(angular+1)};
  });
}

extern "C" void openmx_core_pair_bessel_statistics(const OpenMXPairBesselWorkspace* workspace,
    long* hits, long* misses)
{
  const auto statistics = workspace ? workspace->implementation.statistics() : openmx::core::PairBesselStatistics{};
  if (hits) *hits = statistics.hits;
  if (misses) *misses = statistics.misses;
}

extern "C" void openmx_core_pair_bessel_destroy(OpenMXPairBesselWorkspace* workspace)
{
  delete workspace;
}

extern "C" OpenMXOverlapWorkspace* openmx_core_overlap_create(const OpenMXOverlapShape* shape,
    OpenMXOverlapBuffers* buffers, char* error, std::size_t capacity)
{
  OpenMXOverlapWorkspace* result = nullptr;
  checked_call(error,capacity,[&] {
    if (!shape || !buffers) throw std::invalid_argument("missing overlap shape or output views");
    auto owner = std::make_unique<OpenMXOverlapWorkspace>(openmx::core::OverlapShape{
        shape->maximum_angular,shape->multiplicity,shape->points,shape->kmin,shape->kmax,
        shape->bessel_cache_limit});
    auto& work = owner->implementation;
    OpenMXOverlapBuffers views{};
    views.multiplicity = shape->multiplicity;
    views.points = shape->points;
    views.max_combinations = static_cast<int>(work.offsets().extent(1));
    views.orbital_capacity = work.angular().extent(1);
    views.magnetic_capacity = work.conversion().extent(1);
    views.radial_elements = work.radial().extent(1)*work.radial().extent(2);
    views.sum_elements = work.sums().extent(1)*work.sums().extent(2);
    views.wavevectors = work.wavevectors().data();
    for (int channel = 0; channel < 2; ++channel) {
      views.radial[channel] = work.radial().data_handle()+channel*views.radial_elements;
      views.offsets[channel] = work.offsets().data_handle()+channel*views.max_combinations;
    }
    for (int channel = 0; channel < 4; ++channel)
      views.sums[channel] = work.sums().data_handle()+channel*views.sum_elements;
    for (int channel = 0; channel < 8; ++channel) {
      views.angular[channel] = work.angular().data_handle()+channel*views.orbital_capacity*views.orbital_capacity;
      views.conversion[channel] = work.conversion().data_handle()+channel*views.magnetic_capacity*views.magnetic_capacity;
    }
    *buffers = views;
    result = owner.release();
  });
  return result;
}

extern "C" int openmx_core_overlap_sample(OpenMXOverlapWorkspace* workspace,
    double distance, int angular, OpenMXPairBesselTable* table, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!workspace || !table) throw std::invalid_argument("missing overlap workspace or Bessel output");
    const auto result = workspace->implementation.sample(distance,angular);
    *table = {result.values.data_handle(),&openmx::core::element(result.values,1,0,0),
              result.values.extent(2),static_cast<std::size_t>(angular+1)};
  });
}

extern "C" void openmx_core_overlap_statistics(const OpenMXOverlapWorkspace* workspace,
    long* hits, long* misses)
{
  const auto statistics = workspace ? workspace->implementation.statistics() : openmx::core::PairBesselStatistics{};
  if (hits) *hits = statistics.hits;
  if (misses) *misses = statistics.misses;
}

extern "C" void openmx_core_overlap_destroy(OpenMXOverlapWorkspace* workspace)
{
  delete workspace;
}

extern "C" OpenMXNonlocalWorkspace* openmx_core_nonlocal_create(const OpenMXNonlocalShape* shape,
    OpenMXNonlocalBuffers* buffers, char* error, std::size_t capacity)
{
  OpenMXNonlocalWorkspace* result = nullptr;
  checked_call(error,capacity,[&] {
    if (!shape || !buffers || shape->species <= 0 || !shape->projector_counts ||
        shape->points < 0 || (shape->points && !shape->wavevectors))
      throw std::invalid_argument("invalid nonlocal shape, views or radial grid");
    std::vector<openmx::core::NonlocalSpecies> species;
    for (int s = 0; s < shape->species; ++s) {
      const int count = shape->projector_counts[s];
      if (count < 0 || (count && (!shape->projector_angular || !shape->projector_angular[s])))
        throw std::invalid_argument("invalid nonlocal projector input row");
      species.push_back({count ? std::span<const int>(shape->projector_angular[s]+1,
          static_cast<std::size_t>(count)) : std::span<const int>{}});
    }
    const auto grid = shape->points ? std::span<const double>(shape->wavevectors,
        static_cast<std::size_t>(shape->points)) : std::span<const double>{};
    auto owner = std::make_unique<OpenMXNonlocalWorkspace>(openmx::core::NonlocalShape{
        shape->maximum_angular,shape->multiplicity,shape->projector_maximum_angular},
        species,grid,shape->bessel_cache_limit);
    auto& work = owner->implementation;
    const auto& layout = work.projectors();
    OpenMXNonlocalBuffers views{};
    views.multiplicity = shape->multiplicity;
    views.projector_slots = static_cast<int>(layout.slots());
    views.orbital_capacity = work.angular().extent(1);
    views.projector_capacity = work.angular().extent(2);
    views.conversion_columns = work.conversion().extent(2);
    views.projector_counts = layout.counts().data();
    views.projector_angular = layout.angular().data();
    views.species_offsets = layout.species_offsets().data();
    views.projector_offsets = layout.offsets().data();
    for (int channel = 0; channel < 4; ++channel) {
      views.angular[channel] = work.angular().data_handle()+channel*views.orbital_capacity*views.projector_capacity;
      views.conversion[channel] = work.conversion().data_handle()
          +channel*work.conversion().extent(1)*views.conversion_columns;
    }
    if (shape->points) {
      views.radial = work.radial().data_handle();
      for (int channel = 0; channel < 2; ++channel) {
        views.sums[channel] = work.sums().data_handle()+channel*work.sums().extent(1)*work.sums().extent(2);
        views.nonlocal_radial[channel] = work.nonlocal_radial().data_handle()
            +channel*work.nonlocal_radial().extent(1)*work.nonlocal_radial().extent(2);
      }
    }
    *buffers = views;
    result = owner.release();
  });
  return result;
}

extern "C" int openmx_core_nonlocal_sample(OpenMXNonlocalWorkspace* workspace, double distance,
    int angular, OpenMXPairBesselTable* table, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!workspace || !table) throw std::invalid_argument("missing nonlocal Bessel workspace or output");
    const auto values = workspace->implementation.sample(distance,angular).values;
    *table = {values.data_handle(),&openmx::core::element(values,1,0,0),
              values.extent(2),static_cast<std::size_t>(angular+1)};
  });
}

extern "C" void openmx_core_nonlocal_destroy(OpenMXNonlocalWorkspace* workspace)
{
  delete workspace;
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
      case OPENMX_RADIAL_NEUTRAL_PROJECTOR: policy = openmx::core::RadialKind::neutral_projector; break;
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

extern "C" int openmx_core_expand_neutral_projectors(const OpenMXNeutralExpansion* input,
    int node_count, const double* nodes, const double* weights, char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input || node_count <= 0 || !nodes || !weights ||
        (input->empty_atom != 0 && input->empty_atom != 1))
      throw std::invalid_argument("invalid neutral-projector expansion description");
    rectangular_basis(input->lmax,input->multiplicity);
    rectangular_basis(input->pao_lmax,input->pao_multiplicity);
    const auto radii = mesh(input->radii,input->mesh_count);
    const auto potential = channel(0,input->potential,input->mesh_count);
    const auto pao_radii = mesh(input->pao_radii,input->pao_mesh_count);
    std::vector<openmx::core::RadialChannel> channels;
    if (!input->pao_values || !input->projectors || !input->energies)
      throw std::invalid_argument("missing neutral-projector expansion tables");
    for (int l = 0; l <= input->pao_lmax; ++l) {
      if (!input->pao_values[l]) throw std::invalid_argument("missing expansion PAO angular block");
      for (int m = 0; m < input->pao_multiplicity; ++m)
        channels.push_back(channel(l,input->pao_values[l][m],input->pao_mesh_count));
    }
    for (int l = 0; l <= input->lmax; ++l) {
      if (!input->projectors[l] || !input->energies[l])
        throw std::invalid_argument("missing expansion output angular block");
      for (int m = 0; m < input->multiplicity; ++m)
        if (!input->projectors[l][m]) throw std::invalid_argument("missing expansion output row");
    }
    const openmx::core::RadialTableView pao(openmx::core::RadialKind::pao,pao_radii,channels);
    const auto result = openmx::core::expand_neutral_projectors(pao,radii,potential.values,
        {input->lmax,input->multiplicity,input->pao_lmax,input->pao_multiplicity,input->empty_atom != 0},
        {nodes,static_cast<std::size_t>(node_count)},{weights,static_cast<std::size_t>(node_count)});
    const auto values = result.values.view();
    const auto energies = result.energies.view();
    // All allocations, validation and computation completed. Inputs may alias
    // output storage; no input will be read after this point.
    for (int l = 0; l <= input->lmax; ++l)
      for (int m = 0; m < input->multiplicity; ++m) {
        std::copy_n(&openmx::core::element(values,l,m,0),radii.size(),input->projectors[l][m]);
        input->energies[l][m] = openmx::core::element(energies,l,m);
      }
  });
}

extern "C" int openmx_core_transform_neutral_projectors(OpenMXCoreContext* context, int device,
    const OpenMXNeutralProjectorTable* input, const OpenMXGaussRadialGrid* grid,
    char* error, std::size_t capacity)
{
  return checked_call(error,capacity,[&] {
    if (!input) throw std::invalid_argument("missing neutral-projector table");
    rectangular_basis(input->lmax,input->multiplicity);
    const auto radii = mesh(input->radii,input->mesh_count);
    auto quadrature = gauss_grid(input->lmax,grid);
    std::vector<openmx::core::RadialChannel> channels;
    std::vector<double*> outputs;
    if (!input->values || !input->output) throw std::invalid_argument("missing neutral-projector tables");
    for (int l = 0; l <= input->lmax; ++l) {
      if (!input->values[l] || !input->output[l])
        throw std::invalid_argument("missing neutral-projector angular block");
      for (int m = 0; m < input->multiplicity; ++m) {
        channels.push_back(channel(l,input->values[l][m],input->mesh_count));
        outputs.push_back(input->output[l][m]);
      }
    }
    const openmx::core::RadialTableView table(openmx::core::RadialKind::neutral_projector,radii,channels);
    transform_table(context,device,table,std::move(quadrature),outputs);
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
    context->products.reset();
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

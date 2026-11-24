#include <openmx/exx/exx.h>

#include <openmx/exx/abf.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/forces.hpp>
#include <openmx/exx/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace openmx::exx;

// What the tables of the species were made from.  The tables are kept as long
// as these inputs do not change.
struct OpenMXExxState {
  struct Species {
    double cutoff = 0.0;
    std::vector<double> mesh;
    std::vector<int> angular;
    std::vector<double> values;
    friend bool operator==(const Species&, const Species&) = default;
  };
  struct Grids {
    int radial_intervals = 0;
    double reciprocal_maximum = 0.0;
    int reciprocal_intervals = 0;
    double abf_threshold = 0.0;
    friend bool operator==(const Grids&, const Grids&) = default;
  };
  struct Tables {
    Grids grids;
    std::vector<Species> inputs;
    ReciprocalGrid reciprocal;
    std::vector<SpeciesTable> species;
  };
  std::optional<Tables> tables;
  // The structure and the options of the last preparation: the derivative
  // with respect to the positions evaluates the integrals of the pairs again.
  Structure structure;
  SystemOptions options;
  std::optional<System> system;
  std::vector<std::size_t> offset;   // first orbital of every atom, and N
  std::optional<BlockLayout> layout; // of the site form; made on request
};

namespace {

void write_error(char* error, std::size_t capacity, const char* message) noexcept
{
  if (error == nullptr || capacity == 0) return;
  const std::size_t length = std::min(std::strlen(message),capacity-1);
  std::memcpy(error,message,length);
  error[length] = '\0';
}

// Runs a function of the C++ interface and turns its exceptions into the
// return value and the message of the C interface.
template <class Function>
int guarded(char* error, std::size_t capacity, Function&& function) noexcept
{
  try {
    std::forward<Function>(function)();
    write_error(error,capacity,"");
    return 0;
  } catch (const std::bad_alloc&) {
    write_error(error,capacity,"out of memory");
    return 2;
  } catch (const std::exception& e) {
    write_error(error,capacity,e.what());
    return 1;
  } catch (...) {
    write_error(error,capacity,"unknown error");
    return 1;
  }
}

std::vector<OpenMXExxState::Species> copy_species(const OpenMXExxGeometry& geometry)
{
  if (geometry.species_count <= 0 || geometry.species_data == nullptr)
    throw std::invalid_argument("the geometry has no species");
  std::vector<OpenMXExxState::Species> inputs(static_cast<std::size_t>(geometry.species_count));
  for (std::size_t s = 0; s < inputs.size(); ++s) {
    const OpenMXExxSpecies& source = geometry.species_data[s];
    if (source.mesh_size < 2 || source.shell_count < 1 || source.mesh == nullptr || source.angular == nullptr ||
        source.values == nullptr)
      throw std::invalid_argument("a species has no radial mesh or no radial functions");
    const std::size_t points = static_cast<std::size_t>(source.mesh_size);
    const std::size_t shells = static_cast<std::size_t>(source.shell_count);
    inputs[s].cutoff = source.cutoff;
    inputs[s].mesh.assign(source.mesh,source.mesh+points);
    inputs[s].angular.assign(source.angular,source.angular+shells);
    inputs[s].values.assign(source.values,source.values+shells*points);
  }
  return inputs;
}

OpenMXExxState::Tables make_tables(const OpenMXExxState::Grids& grids, std::vector<OpenMXExxState::Species> inputs)
{
  if (grids.radial_intervals < 2) throw std::invalid_argument("the radial grid needs at least two intervals");
  OpenMXExxState::Tables tables{grids,std::move(inputs),
                                ReciprocalGrid::uniform(grids.reciprocal_maximum,grids.reciprocal_intervals),{}};
  tables.species.reserve(tables.inputs.size());
  for (const OpenMXExxState::Species& input : tables.inputs) {
    if (!(input.cutoff > 0.0) || !std::isfinite(input.cutoff))
      throw std::invalid_argument("the cutoff radius of a species must be positive");
    const RadialGrid grid{0.0,input.cutoff/grids.radial_intervals,grids.radial_intervals};
    const std::size_t points = input.mesh.size();
    std::vector<MeshFunction> shells;
    for (std::size_t shell = 0; shell < input.angular.size(); ++shell)
      shells.push_back({input.angular[shell],std::span<const double>(input.values).subspan(shell*points,points)});
    std::vector<RadialFunction> orbitals = sample_orbitals(grid,input.mesh,shells);
    AuxiliaryBasis basis = make_auxiliary_basis(grid,orbitals,grids.abf_threshold);
    tables.species.emplace_back(grid,std::move(orbitals),std::move(basis.functions),tables.reciprocal);
  }
  return tables;
}

Structure copy_structure(const OpenMXExxGeometry& geometry)
{
  if (geometry.atom_count <= 0 || geometry.positions == nullptr || geometry.species == nullptr)
    throw std::invalid_argument("the geometry has no atoms");
  Structure structure;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t x = 0; x < 3; ++x) structure.lattice[i][x] = geometry.lattice[i][x];
  const std::size_t atoms = static_cast<std::size_t>(geometry.atom_count);
  structure.positions.resize(atoms);
  structure.species.assign(geometry.species,geometry.species+atoms);
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t x = 0; x < 3; ++x) structure.positions[a][x] = geometry.positions[3*a+x];
  return structure;
}

Kernel interaction(const OpenMXExxOptions& options)
{
  switch (options.kernel) {
    case OPENMX_EXX_KERNEL_SHORT_RANGE: return short_range_kernel(options.kernel_parameter);
    case OPENMX_EXX_KERNEL_TRUNCATED_COULOMB: return truncated_coulomb_kernel(options.kernel_parameter);
    default: throw std::invalid_argument("unknown kernel of the exchange interaction");
  }
}

const System* prepared(const OpenMXExxState* state) noexcept
{
  return state != nullptr && state->system.has_value() ? &*state->system : nullptr;
}

const BlockLayout* layout_of(const OpenMXExxState* state) noexcept
{
  return prepared(state) != nullptr && state->layout.has_value() ? &*state->layout : nullptr;
}

// The system and the layout of a call that works on arrays of the site form.
const BlockLayout& required_layout(const OpenMXExxState* state)
{
  if (prepared(state) == nullptr) throw std::invalid_argument("the site form needs a prepared structure");
  if (!state->layout.has_value()) throw std::invalid_argument("the site form needs openmx_exx_prepare_blocks");
  return *state->layout;
}

int atom_value(const OpenMXExxState* state, int atom, const std::vector<std::size_t> System::*member) noexcept
{
  const System* system = prepared(state);
  if (system == nullptr || atom < 0 || static_cast<std::size_t>(atom) >= (system->*member).size()) return -1;
  return static_cast<int>((system->*member)[static_cast<std::size_t>(atom)]);
}

int neighbour_count(const OpenMXExxState* state, int atom, const std::vector<std::vector<Site>> System::*member) noexcept
{
  const System* system = prepared(state);
  if (system == nullptr || atom < 0 || static_cast<std::size_t>(atom) >= (system->*member).size()) return -1;
  return static_cast<int>((system->*member)[static_cast<std::size_t>(atom)].size());
}

} // namespace

extern "C" {

void openmx_exx_default_options(OpenMXExxOptions* options)
{
  if (options == nullptr) return;
  options->kernel = OPENMX_EXX_KERNEL_SHORT_RANGE;
  options->kernel_parameter = 0.11;
  options->abf_threshold = 1.0e-4;
  options->v_cutoff = 8.0;
  options->radial_intervals = 900;
  options->reciprocal_maximum = 60.0;
  options->reciprocal_intervals = 900;
}

OpenMXExxState* openmx_exx_create(void)
{
  return new (std::nothrow) OpenMXExxState;
}

void openmx_exx_destroy(OpenMXExxState* state)
{
  delete state;
}

int openmx_exx_prepare(OpenMXExxState* state, const OpenMXExxOptions* options, const OpenMXExxGeometry* geometry,
                       char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (state == nullptr || options == nullptr || geometry == nullptr)
      throw std::invalid_argument("openmx_exx_prepare needs a state, options and a geometry");
    const OpenMXExxState::Grids grids{options->radial_intervals,options->reciprocal_maximum,
                                      options->reciprocal_intervals,options->abf_threshold};
    std::vector<OpenMXExxState::Species> inputs = copy_species(*geometry);
    // New tables are made aside, so that a failure leaves the state as it was.
    std::optional<OpenMXExxState::Tables> made;
    if (!state->tables.has_value() || !(state->tables->grids == grids) || !(state->tables->inputs == inputs))
      made.emplace(make_tables(grids,std::move(inputs)));
    const OpenMXExxState::Tables& tables = made.has_value() ? *made : *state->tables;

    SystemOptions system_options;
    system_options.interaction = interaction(*options);
    system_options.v_cutoff = options->v_cutoff;
    Structure structure = copy_structure(*geometry);
    System system = prepare_system(structure,tables.species,tables.reciprocal,system_options);
    std::vector<std::size_t> offset(1,0);
    for (const std::size_t u : system.orbital_count) offset.push_back(offset.back()+u);

    if (made.has_value()) state->tables = std::move(made);
    state->structure = std::move(structure);
    state->options = std::move(system_options);
    state->system = std::move(system);
    state->offset = std::move(offset);
    state->layout.reset();
  });
}

int openmx_exx_atom_count(const OpenMXExxState* state)
{
  const System* system = prepared(state);
  return system != nullptr ? static_cast<int>(system->orbital_count.size()) : -1;
}

int openmx_exx_orbital_count(const OpenMXExxState* state)
{
  return prepared(state) != nullptr ? static_cast<int>(state->offset.back()) : -1;
}

int openmx_exx_atom_orbitals(const OpenMXExxState* state, int atom)
{
  return atom_value(state,atom,&System::orbital_count);
}

int openmx_exx_atom_abfs(const OpenMXExxState* state, int atom)
{
  return atom_value(state,atom,&System::abf_count);
}

int openmx_exx_c_neighbours(const OpenMXExxState* state, int atom)
{
  return neighbour_count(state,atom,&System::c_neighbours);
}

int openmx_exx_v_neighbours(const OpenMXExxState* state, int atom)
{
  return neighbour_count(state,atom,&System::v_neighbours);
}

int openmx_exx_report(const OpenMXExxState* state, OpenMXExxReport* report)
{
  const System* system = prepared(state);
  if (system == nullptr || report == nullptr) return 1;
  report->systems = system->report.systems;
  report->eigen_decompositions = system->report.eigen_decompositions;
  report->dropped = system->report.dropped;
  report->smallest_pivot = system->report.smallest_pivot;
  return 0;
}

int openmx_exx_add_gamma(const OpenMXExxState* state, int first_atom, int last_atom, double factor,
                         const double* density, double* k, const OpenMXExxDenseBackend* backend, char* error,
                         size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System* system = prepared(state);
    if (system == nullptr) throw std::invalid_argument("the exchange matrix needs a prepared structure");
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    const int atoms = static_cast<int>(system->orbital_count.size());
    if (first_atom < 0 || last_atom < first_atom || last_atom > atoms)
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const std::size_t n = state->offset.back();
    add_exchange_gamma_range(*system,static_cast<std::size_t>(first_atom),static_cast<std::size_t>(last_atom),factor,
                             std::span<const double>(density,n*n),std::span<double>(k,n*n),DenseBackend(backend));
  });
}

int openmx_exx_prepare_blocks(OpenMXExxState* state, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (prepared(state) == nullptr) throw std::invalid_argument("the site form needs a prepared structure");
    if (!state->layout.has_value()) state->layout = exchange_layout(*state->system);
  });
}

long openmx_exx_block_count(const OpenMXExxState* state)
{
  const BlockLayout* layout = layout_of(state);
  return layout != nullptr ? static_cast<long>(layout->keys.size()) : -1;
}

size_t openmx_exx_block_elements(const OpenMXExxState* state)
{
  const BlockLayout* layout = layout_of(state);
  return layout != nullptr ? layout->elements() : 0;
}

int openmx_exx_block(const OpenMXExxState* state, long index, int* row_atom, int* column_atom, int cell[3],
                     size_t* offset)
{
  const BlockLayout* layout = layout_of(state);
  if (layout == nullptr || index < 0 || static_cast<std::size_t>(index) >= layout->keys.size()) return 1;
  const BlockKey& key = layout->keys[static_cast<std::size_t>(index)];
  if (row_atom != nullptr) *row_atom = key.row;
  if (column_atom != nullptr) *column_atom = key.column;
  if (cell != nullptr) std::copy(key.cell.begin(),key.cell.end(),cell);
  if (offset != nullptr) *offset = layout->offset[static_cast<std::size_t>(index)];
  return 0;
}

long openmx_exx_find_block(const OpenMXExxState* state, int row_atom, int column_atom, const int cell[3])
{
  const BlockLayout* layout = layout_of(state);
  if (layout == nullptr || cell == nullptr) return -1;
  const std::size_t index = layout->find(BlockKey{row_atom,column_atom,{cell[0],cell[1],cell[2]}});
  return index < layout->keys.size() ? static_cast<long>(index) : -1;
}

int openmx_exx_add_blocks(const OpenMXExxState* state, int first_atom, int last_atom, double factor,
                          const double* density, double* k, const OpenMXExxDenseBackend* backend, char* error,
                          size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    const int atoms = static_cast<int>(system.orbital_count.size());
    if (first_atom < 0 || last_atom < first_atom || last_atom > atoms)
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const std::size_t elements = layout.elements();
    add_exchange_blocks_range(system,layout,static_cast<std::size_t>(first_atom),static_cast<std::size_t>(last_atom),
                              factor,std::span<const double>(density,elements),std::span<double>(k,elements),
                              DenseBackend(backend));
  });
}

int openmx_exx_add_position_gradient_gamma(const OpenMXExxState* state, int first_atom, int last_atom, double factor,
                                           const double* density, double* gradient,
                                           const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System* system = prepared(state);
    if (system == nullptr) throw std::invalid_argument("the derivative with respect to the positions needs a prepared structure");
    if (density == nullptr || gradient == nullptr)
      throw std::invalid_argument("the derivative with respect to the positions needs a density matrix and an output");
    const std::size_t atoms = system->orbital_count.size();
    if (first_atom < 0 || last_atom < first_atom || last_atom > static_cast<int>(atoms))
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const std::size_t n = state->offset.back();
    const DenseBackend products(backend);
    const PositionGradient derivative(state->structure,state->tables->species,state->tables->reciprocal,state->options,
                                      *system);
    // Summed aside, so that a failure adds nothing.
    std::vector<double> sum(3*atoms,0.0);
    for (int atom = first_atom; atom < last_atom; ++atom)
      derivative.add_gamma(static_cast<std::size_t>(atom),factor,std::span<const double>(density,n*n),sum,products);
    for (std::size_t e = 0; e < sum.size(); ++e) gradient[e] += sum[e];
  });
}

int openmx_exx_add_position_gradient_blocks(const OpenMXExxState* state, int first_atom, int last_atom, double factor,
                                            const double* density, double* gradient,
                                            const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    if (density == nullptr || gradient == nullptr)
      throw std::invalid_argument("the derivative with respect to the positions needs a density matrix and an output");
    const std::size_t atoms = system.orbital_count.size();
    if (first_atom < 0 || last_atom < first_atom || last_atom > static_cast<int>(atoms))
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const DenseBackend products(backend);
    const PositionGradient derivative(state->structure,state->tables->species,state->tables->reciprocal,state->options,
                                      system);
    std::vector<double> sum(3*atoms,0.0);
    for (int atom = first_atom; atom < last_atom; ++atom)
      derivative.add_blocks(layout,static_cast<std::size_t>(atom),factor,
                            std::span<const double>(density,layout.elements()),sum,products);
    for (std::size_t e = 0; e < sum.size(); ++e) gradient[e] += sum[e];
  });
}

int openmx_exx_symmetrize_blocks(const OpenMXExxState* state, double* values, double* asymmetry, char* error,
                                 size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    if (values == nullptr) throw std::invalid_argument("no array to make symmetric");
    const std::span<double> array(values,layout.elements());
    const double before = block_asymmetry(*state->system,layout,array);
    symmetrize_blocks(*state->system,layout,array);
    if (asymmetry != nullptr) *asymmetry = before;
  });
}

double openmx_exx_energy_gamma(size_t count, const double* density, const double* k)
{
  if (density == nullptr || k == nullptr) return 0.0;
  double total = 0.0;
  for (size_t e = 0; e < count; ++e) total += density[e]*k[e];
  return 0.5*total;
}

double openmx_exx_change_gamma(size_t count, const double* density, const double* k, const double* k_previous)
{
  if (density == nullptr || k == nullptr || k_previous == nullptr) return 0.0;
  double total = 0.0;
  for (size_t e = 0; e < count; ++e) total += density[e]*(k[e]-k_previous[e]);
  return total;
}

} // extern "C"

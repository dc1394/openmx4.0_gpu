#include <openmx/exx/exx.h>

#include <openmx/exx/abf.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/forces.hpp>
#include <openmx/exx/system.hpp>

#include <algorithm>
#include <array>
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
  OpenMXExxAddRows host_add_rows = nullptr;
  double minimum_device_work = 0.0;
  // After openmx_exx_prepare_part with several parts: the atoms whose blocks
  // openmx_exx_set_atom_tables has not replaced by their sums yet.  The
  // system serves the builds and the derivatives when none is left.
  std::vector<unsigned char> partial;
  std::size_t partial_count = 0;
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

// The system of a call that reads the T and V blocks: the blocks of a divided
// preparation must have been added up.
const System& complete(const OpenMXExxState* state, const char* missing)
{
  const System* system = prepared(state);
  if (system == nullptr) throw std::invalid_argument(missing);
  if (state->partial_count != 0)
    throw std::invalid_argument("the blocks of a divided preparation have not been added up for every atom");
  return *system;
}

void prepare(OpenMXExxState* state, const OpenMXExxOptions* options, const OpenMXExxGeometry* geometry,
             const PrepareShare& share, const DenseBackend& backend = DenseBackend());

const BlockLayout* layout_of(const OpenMXExxState* state) noexcept
{
  return prepared(state) != nullptr && state->layout.has_value() ? &*state->layout : nullptr;
}

// The system and the layout of a call that works on arrays of the site form.
const BlockLayout& required_layout(const OpenMXExxState* state)
{
  complete(state,"the site form needs a prepared structure");
  if (!state->layout.has_value()) throw std::invalid_argument("the site form needs openmx_exx_prepare_blocks");
  return *state->layout;
}

void prepare(OpenMXExxState* state, const OpenMXExxOptions* options, const OpenMXExxGeometry* geometry,
             const PrepareShare& share, const DenseBackend& backend)
{
  if (state == nullptr || options == nullptr || geometry == nullptr)
    throw std::invalid_argument("the preparation needs a state, options and a geometry");
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
  System system = prepare_system(structure,tables.species,tables.reciprocal,system_options,share,backend);
  std::vector<std::size_t> offset(1,0);
  for (const std::size_t u : system.orbital_count) offset.push_back(offset.back()+u);
  std::vector<unsigned char> partial(share.parts > 1 ? system.orbital_count.size() : 0,1);

  if (made.has_value()) state->tables = std::move(made);
  state->structure = std::move(structure);
  state->options = std::move(system_options);
  state->system = std::move(system);
  state->offset = std::move(offset);
  state->layout.reset();
  state->partial_count = partial.size();
  state->partial = std::move(partial);
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

// The arrays of the weights of the atoms that a range of ABFs touches, for
// add_block_weights_*_abfs(); the entries of the other atoms are not read.
std::vector<std::span<double>> weight_arrays(const System& system, long first_abf, long last_abf,
                                             const double* density, double* const* weights, bool half = false)
{
  if (density == nullptr || weights == nullptr)
    throw std::invalid_argument("the weights of the blocks need a density matrix and arrays for the atoms");
  if (first_abf < 0 || last_abf < first_abf) throw std::out_of_range("the range of ABFs is not inside the unit cell");
  const std::size_t atoms = system.orbital_count.size();
  const std::size_t first = static_cast<std::size_t>(first_abf), last = static_cast<std::size_t>(last_abf);
  std::vector<std::span<double>> arrays(atoms);
  std::size_t offset = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t next = offset+system.abf_count[a];
    // The half form adds to the T blocks of the V neighbours as well.
    if (half ? first < last : std::max(first,offset) < std::min(last,next)) {
      if (weights[a] == nullptr) throw std::invalid_argument("an atom of the range has no array for its weights");
      arrays[a] = std::span<double>(weights[a],table_elements(system,a));
    }
    offset = next;
  }
  return arrays;
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

void openmx_exx_set_host_add_rows(OpenMXExxState* state, OpenMXExxAddRows add_rows)
{
  if (state != nullptr) state->host_add_rows = add_rows;
}

void openmx_exx_set_device_minimum_work(OpenMXExxState* state, double minimum)
{
  if (state != nullptr) state->minimum_device_work = minimum >= 0.0 ? minimum : 0.0;
}

int openmx_exx_prepare(OpenMXExxState* state, const OpenMXExxOptions* options, const OpenMXExxGeometry* geometry,
                       char* error, size_t capacity)
{
  return guarded(error,capacity,[&] { prepare(state,options,geometry,PrepareShare{}); });
}

int openmx_exx_prepare_part(OpenMXExxState* state, const OpenMXExxOptions* options, const OpenMXExxGeometry* geometry,
                            int part, int parts, char* error, size_t capacity)
{
  return openmx_exx_prepare_part_pieces(state,options,geometry,part,parts,nullptr,error,capacity);
}

int openmx_exx_prepare_part_pieces(OpenMXExxState* state, const OpenMXExxOptions* options,
                                   const OpenMXExxGeometry* geometry, int part, int parts,
                                   const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (parts < 1 || part < 0 || part >= parts)
      throw std::invalid_argument("the part of a divided preparation is out of range");
    prepare(state,options,geometry,PrepareShare{static_cast<std::size_t>(part),static_cast<std::size_t>(parts)},
            DenseBackend(backend));
  });
}

size_t openmx_exx_atom_table_elements(const OpenMXExxState* state, int atom)
{
  const System* system = prepared(state);
  if (system == nullptr || atom < 0 || static_cast<std::size_t>(atom) >= system->orbital_count.size()) return 0;
  try {
    return table_elements(*system,static_cast<std::size_t>(atom));
  } catch (...) {
    return 0;
  }
}

int openmx_exx_get_atom_tables(const OpenMXExxState* state, int atom, double* values)
{
  const System* system = prepared(state);
  if (system == nullptr || values == nullptr || atom < 0 ||
      static_cast<std::size_t>(atom) >= system->orbital_count.size())
    return 1;
  const std::size_t a = static_cast<std::size_t>(atom);
  try {
    get_tables(*system,a,std::span<double>(values,table_elements(*system,a)));
  } catch (...) {
    return 1;
  }
  return 0;
}

int openmx_exx_set_atom_tables(OpenMXExxState* state, int atom, const double* values)
{
  if (prepared(state) == nullptr || values == nullptr || atom < 0 ||
      static_cast<std::size_t>(atom) >= state->system->orbital_count.size())
    return 1;
  const std::size_t a = static_cast<std::size_t>(atom);
  try {
    set_tables(*state->system,a,std::span<const double>(values,table_elements(*state->system,a)));
  } catch (...) {
    return 1;
  }
  if (a < state->partial.size() && state->partial[a]) {
    state->partial[a] = 0;
    --state->partial_count;
  }
  return 0;
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

int openmx_exx_prepared_v_neighbours(const OpenMXExxState* state, int atom)
{
  const System* system = prepared(state);
  if (system == nullptr || system->folding.prepared.empty()) return neighbour_count(state,atom,&System::v_neighbours);
  if (atom < 0 || static_cast<std::size_t>(atom) >= system->folding.prepared.size()) return -1;
  return static_cast<int>(system->folding.prepared[static_cast<std::size_t>(atom)].size());
}

int openmx_exx_fold_v_neighbours(OpenMXExxState* state, const int period[3], char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (period == nullptr) throw std::invalid_argument("the folding needs a period");
    complete(state,"the folding needs a prepared structure");
    // The call changes nothing when it fails.
    fold_v_neighbours(*state->system,state->structure,{period[0],period[1],period[2]});
    state->layout.reset();
  });
}

int openmx_exx_fold_columns(OpenMXExxState* state, const int antiperiodic[3], char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (antiperiodic == nullptr) throw std::invalid_argument("the folding of the columns needs its antiperiodic directions");
    complete(state,"the folding of the columns needs a prepared structure");
    // The call changes nothing when it fails; the layout does not depend on it.
    fold_columns(*state->system,{antiperiodic[0] != 0,antiperiodic[1] != 0,antiperiodic[2] != 0});
  });
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
    const System& system = complete(state,"the exchange matrix needs a prepared structure");
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    const int atoms = static_cast<int>(system.orbital_count.size());
    if (first_atom < 0 || last_atom < first_atom || last_atom > atoms)
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const std::size_t n = state->offset.back();
    add_exchange_gamma_range(system,static_cast<std::size_t>(first_atom),static_cast<std::size_t>(last_atom),factor,
                             std::span<const double>(density,n*n),std::span<double>(k,n*n),DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

long openmx_exx_abf_count(const OpenMXExxState* state)
{
  const System* system = prepared(state);
  if (system == nullptr) return -1;
  std::size_t total = 0;
  for (const std::size_t n : system->abf_count) total += n;
  return static_cast<long>(total);
}

int openmx_exx_divide_abfs(const OpenMXExxState* state, int sites, int half, int processes, long* boundaries)
{
  const System* system = prepared(state);
  if (system == nullptr || boundaries == nullptr || processes < 1) return 1;
  try {
    const std::vector<std::size_t> values =
        divide_abfs(*system,sites != 0,half != 0,static_cast<std::size_t>(processes),layout_of(state));
    for (std::size_t r = 0; r < values.size(); ++r) boundaries[r] = static_cast<long>(values[r]);
  } catch (...) {
    return 1;
  }
  return 0;
}

int openmx_exx_add_gamma_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                              const double* density, double* k, const OpenMXExxDenseBackend* backend, char* error,
                              size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the exchange matrix needs a prepared structure");
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    if (first_abf < 0 || last_abf < first_abf) throw std::out_of_range("the range of ABFs is not inside the unit cell");
    const std::size_t n = state->offset.back();
    add_exchange_gamma_abfs(system,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),factor,
                            std::span<const double>(density,n*n),std::span<double>(k,n*n),DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_prepare_blocks(OpenMXExxState* state, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    if (prepared(state) == nullptr) throw std::invalid_argument("the site form needs a prepared structure");
    if (!state->layout.has_value()) state->layout = exchange_layout(*state->system);
  });
}

int openmx_exx_prepare_projected_blocks(OpenMXExxState* state, size_t count,
    const OpenMXExxBlockKey* keys, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System* system = prepared(state);
    if (system == nullptr || !keys)
      throw std::invalid_argument("the projected layout needs a prepared unfolded structure and block keys");
    std::vector<BlockKey> blocks;
    blocks.reserve(count);
    for (const auto& key : std::span(keys,count))
      blocks.push_back({key.row_atom,key.column_atom,{key.cell[0],key.cell[1],key.cell[2]}});
    state->layout = projected_exchange_layout(*system,std::move(blocks));
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
                              DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_blocks_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                               const double* density, double* k, const OpenMXExxDenseBackend* backend, char* error,
                               size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    if (first_abf < 0 || last_abf < first_abf) throw std::out_of_range("the range of ABFs is not inside the unit cell");
    const std::size_t elements = layout.elements();
    add_exchange_blocks_abfs(system,layout,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),
                             factor,std::span<const double>(density,elements),std::span<double>(k,elements),
                             DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_position_gradient_gamma(const OpenMXExxState* state, int first_atom, int last_atom, double factor,
                                           const double* density, double* gradient,
                                           const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System* system = &complete(state,"the derivative with respect to the positions needs a prepared structure");
    if (density == nullptr || gradient == nullptr)
      throw std::invalid_argument("the derivative with respect to the positions needs a density matrix and an output");
    const std::size_t atoms = system->orbital_count.size();
    if (first_atom < 0 || last_atom < first_atom || last_atom > static_cast<int>(atoms))
      throw std::out_of_range("the range of centre atoms is not inside the unit cell");
    const std::size_t n = state->offset.back();
    const DenseBackend products(backend,state->host_add_rows);
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
    const DenseBackend products(backend,state->host_add_rows);
    const PositionGradient derivative(state->structure,state->tables->species,state->tables->reciprocal,state->options,
                                      system);
    std::vector<double> sum(3*atoms,0.0);
    for (int atom = first_atom; atom < last_atom; ++atom)
      derivative.add_blocks(layout,static_cast<std::size_t>(atom),factor,
                            std::span<const double>(density,layout.elements()),sum,products);
    for (std::size_t e = 0; e < sum.size(); ++e) gradient[e] += sum[e];
  });
}

int openmx_exx_add_block_weights_gamma_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                                            const double* density, double* const* weights,
                                            const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the weights of the blocks need a prepared structure");
    const std::vector<std::span<double>> arrays = weight_arrays(system,first_abf,last_abf,density,weights);
    const std::size_t n = state->offset.back();
    add_block_weights_gamma_abfs(system,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),factor,
                                 std::span<const double>(density,n*n),arrays,DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_block_weights_blocks_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                                             const double* density, double* const* weights,
                                             const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    const std::vector<std::span<double>> arrays = weight_arrays(system,first_abf,last_abf,density,weights);
    add_block_weights_blocks_abfs(system,layout,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),
                                  factor,std::span<const double>(density,layout.elements()),arrays,
                                  DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_block_weights_gamma_half_abfs(const OpenMXExxState* state, long first_abf, long last_abf,
                                                 double factor, const double* density, double* const* weights,
                                                 const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the weights of the blocks need a prepared structure");
    const std::vector<std::span<double>> arrays = weight_arrays(system,first_abf,last_abf,density,weights,true);
    const std::size_t n = state->offset.back();
    add_block_weights_gamma_half_abfs(system,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),
                                      factor,std::span<const double>(density,n*n),arrays,DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_block_weights_blocks_half_abfs(const OpenMXExxState* state, long first_abf, long last_abf,
                                                  double factor, const double* density, double* const* weights,
                                                  const OpenMXExxDenseBackend* backend, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    const std::vector<std::span<double>> arrays = weight_arrays(system,first_abf,last_abf,density,weights,true);
    add_block_weights_blocks_half_abfs(system,layout,static_cast<std::size_t>(first_abf),
                                       static_cast<std::size_t>(last_abf),factor,
                                       std::span<const double>(density,layout.elements()),arrays,DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_position_gradient_pairs(const OpenMXExxState* state, const double* const* weights, int part,
                                           int parts, double* gradient, char* error, size_t capacity)
{
  return openmx_exx_add_position_gradient_pairs_pieces(state,weights,part,parts,nullptr,gradient,error,capacity);
}

int openmx_exx_add_position_gradient_pairs_pieces(const OpenMXExxState* state, const double* const* weights,
                                                  int part, int parts, const OpenMXExxDenseBackend* backend,
                                                  double* gradient, char* error, size_t capacity)
{
  return openmx_exx_add_geometry_gradient_pairs(state,weights,part,parts,backend,nullptr,
                                                 gradient,nullptr,0,error,capacity);
}

int openmx_exx_add_geometry_gradient_pairs(const OpenMXExxState* state, const double* const* weights,
    int part, int parts, const OpenMXExxDenseBackend* backend, OpenMXExxDot dot,
    double* gradient, double* strain, int volume_scaled_cutoff, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the derivative with respect to the positions needs a prepared structure");
    if (weights == nullptr || gradient == nullptr)
      throw std::invalid_argument("the derivative with respect to the positions needs the weights of the blocks and an output");
    if (parts < 1 || part < 0 || part >= parts)
      throw std::invalid_argument("the part of a divided derivative is out of range");
    const std::size_t atoms = system.orbital_count.size();
    std::vector<std::span<const double>> arrays(atoms);
    for (std::size_t a = 0; a < atoms; ++a) {
      if (weights[a] == nullptr) throw std::invalid_argument("an atom has no array of weights");
      arrays[a] = std::span<const double>(weights[a],table_elements(system,a));
    }
    const PositionGradient derivative(state->structure,state->tables->species,state->tables->reciprocal,state->options,
                                      system);
    // Summed aside, so that a failure adds nothing.
    std::vector<double> sum(3*atoms,0.0);
    std::array<double,9> cell{};
    derivative.add_pairs(arrays,PrepareShare{static_cast<std::size_t>(part),static_cast<std::size_t>(parts)},sum,
                         DenseBackend(backend,state->host_add_rows,state->minimum_device_work),
                         strain ? std::span<double>(cell) : std::span<double>{}, volume_scaled_cutoff!=0, dot);
    for (std::size_t e = 0; e < sum.size(); ++e) gradient[e] += sum[e];
    if (strain)
      for (std::size_t e = 0; e < cell.size(); ++e) strain[e] += cell[e];
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

int openmx_exx_add_gamma_half_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                                   const double* density, double* k, const OpenMXExxDenseBackend* backend,
                                   char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the exchange matrix needs a prepared structure");
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    if (first_abf < 0 || last_abf < first_abf) throw std::out_of_range("the range of ABFs is not inside the unit cell");
    const std::size_t n = state->offset.back();
    add_exchange_gamma_half_abfs(system,static_cast<std::size_t>(first_abf),static_cast<std::size_t>(last_abf),
                                 factor,std::span<const double>(density,n*n),std::span<double>(k,n*n),
                                 DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_add_blocks_half_abfs(const OpenMXExxState* state, long first_abf, long last_abf, double factor,
                                    const double* density, double* k, const OpenMXExxDenseBackend* backend,
                                    char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    const System& system = *state->system;
    if (density == nullptr || k == nullptr) throw std::invalid_argument("the exchange matrix needs a density matrix and an output");
    if (first_abf < 0 || last_abf < first_abf) throw std::out_of_range("the range of ABFs is not inside the unit cell");
    const std::size_t elements = layout.elements();
    add_exchange_blocks_half_abfs(system,layout,static_cast<std::size_t>(first_abf),
                                  static_cast<std::size_t>(last_abf),factor,std::span<const double>(density,elements),
                                  std::span<double>(k,elements),DenseBackend(backend,state->host_add_rows,state->minimum_device_work));
  });
}

int openmx_exx_mirror_gamma(const OpenMXExxState* state, double* values, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const System& system = complete(state,"the mirror image needs a prepared structure");
    if (values == nullptr) throw std::invalid_argument("no array to add the mirror image to");
    const std::size_t n = state->offset.back();
    mirror_gamma(system,std::span<double>(values,n*n));
  });
}

int openmx_exx_mirror_blocks(const OpenMXExxState* state, double* values, char* error, size_t capacity)
{
  return guarded(error,capacity,[&] {
    const BlockLayout& layout = required_layout(state);
    if (values == nullptr) throw std::invalid_argument("no array to add the mirror image to");
    mirror_blocks(*state->system,layout,std::span<double>(values,layout.elements()));
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

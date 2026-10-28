#ifndef OPENMX_IO_SPECIES_HPP
#define OPENMX_IO_SPECIES_HPP

// The "<Definition.of.Atomic.Species" block of an OpenMX input: one line per
// species with the species name, the basis specification (see
// basis_spec.hpp), the pseudopotential name and an optional atomic mass.

#include "keyword_file.hpp"

#include <optional>
#include <string>
#include <vector>

namespace openmx::io {

struct SpeciesDefinition {
  std::string name, basis, pseudopotential;
  double mass = 0.0;
  bool mass_given = false;
};

// Reads the block from `file`.  nullopt with an empty `error` when the block
// is absent (the original reader then leaves the species untouched), and
// nullopt with the original reader's message when the block does not hold
// exactly `expected` lines of at least three tokens.
std::optional<std::vector<SpeciesDefinition>> read_species_definitions(KeywordFile& file, int expected,
                                                                       std::string* error = nullptr);

}  // namespace openmx::io

// The top file of a C stack (c_api.cpp), for the C entries of this header.
struct OpenMXKeywordFiles;
openmx::io::KeywordFile* openmx_io_keyword_files_current(OpenMXKeywordFiles* files) noexcept;

#endif

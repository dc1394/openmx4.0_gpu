#ifndef OPENMX_IO_ATOMS_HPP
#define OPENMX_IO_ATOMS_HPP

// The "<Atoms.SpeciesAndCoordinates" block of an OpenMX input: one line per
// atom with the sequential number, the species name, three coordinates and
// the initial up/down spin charges; non-collinear inputs add the spin and
// orbital Euler angles (degrees) and a constraint flag; an orbital
// polarization keyword may follow.  Like the original reader's sscanf, a
// line may stop early: `fields` counts the converted items and the rest
// keep their defaults.

#include "keyword_file.hpp"

#include <optional>
#include <string>
#include <vector>

namespace openmx::io {

struct AtomDefinition {
  int fields = 0;                 // converted items (index, species, x, y, z, up, down, [angles, flag,] polarization)
  int index = 0;
  std::string species;
  double position[3] = {0.0, 0.0, 0.0};
  double spin[2] = {0.0, 0.0};    // initial up and down charges
  double spin_angle[2] = {0.0, 0.0};      // theta, phi in degrees (non-collinear)
  double orbital_angle[2] = {0.0, 0.0};   // theta, phi in degrees (non-collinear)
  int constraint = 0;             // spin-angle constraint flag (non-collinear)
  std::string orbital_polarization;
};

// nullopt with an empty `error` when the block is absent; nullopt with the
// original reader's message when the block does not hold exactly `expected`
// lines or is unterminated.
std::optional<std::vector<AtomDefinition>> read_atom_definitions(KeywordFile& file, int expected, bool noncollinear,
                                                                 std::string* error = nullptr);

}  // namespace openmx::io

#endif

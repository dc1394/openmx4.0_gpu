#ifndef OPENMX_IO_SETTINGS_HPP
#define OPENMX_IO_SETTINGS_HPP

// Typed settings of an OpenMX input: the structural keywords and blocks read
// with the defaults and spellings of the application's Input_std.c, without
// its global tables.  read_settings composes the keyword, species, atom and
// unit-cell readers; the remaining scalar keywords of the input are reached
// through KeywordFile until they migrate here.

#include "atoms.hpp"
#include "keyword_file.hpp"
#include "species.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace openmx::io {

enum class LengthUnit { angstrom = 0, bohr = 1, fractional = 2 };
enum class SpinPolarization { off = 0, collinear = 1, noncollinear = 3 };

struct Settings {
  std::string system_name = "default";          // System.Name
  std::string current_directory = "./";         // System.CurrrentDirectory (the legacy spelling)
  std::string data_path = "../DFT_DATA19";      // DATA.PATH
  int stdout_level = 1;                         // level.of.stdout
  SpinPolarization spin_polarization = SpinPolarization::off;   // scf.SpinPolarization
  int species_count = 0;                        // Species.Number
  std::vector<SpeciesDefinition> species;       // <Definition.of.Atomic.Species (empty when absent)
  int atom_count = 0;                           // Atoms.Number
  LengthUnit coordinates_unit = LengthUnit::angstrom;   // Atoms.SpeciesAndCoordinates.Unit
  std::vector<AtomDefinition> atoms;            // <Atoms.SpeciesAndCoordinates (empty when absent)
  LengthUnit unit_vectors_unit = LengthUnit::angstrom;  // Atoms.UnitVectors.Unit (Ang or AU)
  bool unit_vectors_given = false;              // <Atoms.Unitvectors present
  std::array<std::array<double, 3>, 3> unit_vectors{};  // rows a1, a2, a3 in the unit above
};

// The unit-cell block "<Atoms.Unitvectors": nine numbers in reading order,
// the ninth on the block's last line (what the original fscanf/closing-tag
// sequence accepted).  nullopt with an empty `error` when absent.
std::optional<std::array<std::array<double, 3>, 3>> read_unit_vectors(KeywordFile& file, std::string* error = nullptr);

// nullopt with the first reader's message on a malformed block or an invalid
// keyword value; absent blocks leave their vectors empty.
std::optional<Settings> read_settings(KeywordFile& file, std::string* error = nullptr);

}  // namespace openmx::io

#endif

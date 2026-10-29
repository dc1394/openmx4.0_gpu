#include <openmx/io/settings.h>
#include <openmx/io/settings.hpp>

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace openmx::io {

std::optional<std::array<std::array<double, 3>, 3>> read_unit_vectors(KeywordFile& file, std::string* error)
{
  static constexpr const char* format_error = "Format error for Atoms.Unitvectors\n";
  if (error) error->clear();
  const std::optional<std::vector<std::string>> lines = file.block("Atoms.Unitvectors");
  if (!lines) {
    if (file.find("<Atoms.Unitvectors") && error) *error = format_error;
    return std::nullopt;
  }
  std::array<std::array<double, 3>, 3> vectors{};
  int read = 0;
  std::size_t last_line = 0;
  for (std::size_t i = 0; i < lines->size() && read < 9; ++i) {
    std::istringstream in((*lines)[i]);
    std::string token;
    while (read < 9 && in >> token) {
      char* end = nullptr;
      const double value = std::strtod(token.c_str(), &end);
      if (end == token.c_str()) { if (error) *error = format_error; return std::nullopt; }
      vectors[static_cast<std::size_t>(read / 3)][static_cast<std::size_t>(read % 3)] = value;
      ++read;
      last_line = i;
    }
  }
  if (read < 9 || last_line + 1 != lines->size()) {
    if (error) *error = format_error;
    return std::nullopt;
  }
  return vectors;
}

std::optional<Settings> read_settings(KeywordFile& file, std::string* error)
{
  if (error) error->clear();
  Settings s;
  file.string("System.CurrrentDirectory", s.current_directory, "./");
  file.string("System.Name", s.system_name, "default");
  file.string("DATA.PATH", s.data_path, "../DFT_DATA19");
  file.integer("level.of.stdout", s.stdout_level, 1);
  int value = 0;
  static constexpr Choice spin_choices[] = {{"off", 0}, {"on", 1}, {"nc", 3}};
  if (file.choice("scf.SpinPolarization", value, spin_choices) == Lookup::invalid) {
    if (error) *error = "scf.SpinPolarization: unknown value\n";
    return std::nullopt;
  }
  s.spin_polarization = static_cast<SpinPolarization>(value);
  file.integer("Species.Number", s.species_count, 0);
  static constexpr Choice coordinate_units[] = {{"Ang", 0}, {"AU", 1}, {"FRAC", 2}};
  if (file.choice("Atoms.SpeciesAndCoordinates.Unit", value, coordinate_units) == Lookup::invalid) {
    if (error) *error = "Atoms.SpeciesAndCoordinates.Unit: unknown value\n";
    return std::nullopt;
  }
  s.coordinates_unit = static_cast<LengthUnit>(value);
  static constexpr Choice cell_units[] = {{"Ang", 0}, {"AU", 1}};
  if (file.choice("Atoms.UnitVectors.Unit", value, cell_units) == Lookup::invalid) {
    if (error) *error = "Atoms.UnitVectors.Unit: unknown value\n";
    return std::nullopt;
  }
  s.unit_vectors_unit = static_cast<LengthUnit>(value);
  file.integer("Atoms.Number", s.atom_count, 0);

  std::string message;
  if (auto species = read_species_definitions(file, s.species_count, &message)) s.species = std::move(*species);
  else if (!message.empty()) { if (error) *error = message; return std::nullopt; }
  if (auto atoms = read_atom_definitions(file, s.atom_count, s.spin_polarization == SpinPolarization::noncollinear, &message))
    s.atoms = std::move(*atoms);
  else if (!message.empty()) { if (error) *error = message; return std::nullopt; }
  if (auto cell = read_unit_vectors(file, &message)) { s.unit_vectors = *cell; s.unit_vectors_given = true; }
  else if (!message.empty()) { if (error) *error = message; return std::nullopt; }
  return s;
}

}  // namespace openmx::io

extern "C" int openmx_io_read_unit_vectors(OpenMXKeywordFiles* files, double vectors[3][3], char* message, size_t message_capacity)
{
  if (message && message_capacity) message[0] = '\0';
  if (!files || !vectors) return -1;
  openmx::io::KeywordFile* file = openmx_io_keyword_files_current(files);
  if (!file) return -1;
  try {
    std::string error;
    const auto cell = openmx::io::read_unit_vectors(*file, &error);
    if (!cell) {
      if (error.empty()) return 0;
      if (message && message_capacity) std::snprintf(message, message_capacity, "%s", error.c_str());
      return -1;
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) vectors[i][j] = (*cell)[static_cast<size_t>(i)][static_cast<size_t>(j)];
    return 1;
  } catch (...) {
    return -1;
  }
}

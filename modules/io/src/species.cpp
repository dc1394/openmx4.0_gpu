#include <openmx/io/species.h>
#include <openmx/io/species.hpp>

#include <cstdio>
#include <cstring>
#include <sstream>

namespace openmx::io {

std::optional<std::vector<SpeciesDefinition>> read_species_definitions(KeywordFile& file, int expected, std::string* error)
{
  static constexpr const char* format_error = "Format error for Definition.of.Atomic.Species\n";
  if (error) error->clear();
  const std::optional<std::vector<std::string>> lines = file.block("Definition.of.Atomic.Species");
  if (!lines) {
    // an unterminated block is a format error; a missing one is not
    if (file.find("<Definition.of.Atomic.Species") && error) *error = format_error;
    return std::nullopt;
  }
  if (expected < 0 || static_cast<int>(lines->size()) != expected) {
    if (error) *error = format_error;
    return std::nullopt;
  }
  std::vector<SpeciesDefinition> species;
  species.reserve(lines->size());
  for (const std::string& line : *lines) {
    SpeciesDefinition s;
    std::istringstream in(line);
    if (!(in >> s.name >> s.basis >> s.pseudopotential)) {
      if (error) *error = format_error;
      return std::nullopt;
    }
    std::string mass;
    if (in >> mass) {
      // sscanf("%lf") semantics: a leading number is taken, the rest ignored
      char* end = nullptr;
      const double value = std::strtod(mass.c_str(), &end);
      if (end != mass.c_str()) { s.mass = value; s.mass_given = true; }
    }
    species.push_back(std::move(s));
  }
  return species;
}

}  // namespace openmx::io

namespace {
bool store(const std::string& text, char* value, size_t capacity) noexcept
{
  if (!value || text.size() >= capacity) return false;
  std::memcpy(value, text.c_str(), text.size() + 1);
  return true;
}
}  // namespace

extern "C" int openmx_io_read_species_definitions(OpenMXKeywordFiles* files, int count,
                                                  char* const* names, char* const* basis, char* const* pseudopotentials,
                                                  double* mass, size_t capacity, char* message, size_t message_capacity)
{
  if (message && message_capacity) message[0] = '\0';
  if (!files || count < 0 || (count > 0 && (!names || !basis || !pseudopotentials || !mass)) || capacity == 0) return -1;
  openmx::io::KeywordFile* file = openmx_io_keyword_files_current(files);
  if (!file) return -1;
  try {
    std::string error;
    const auto species = openmx::io::read_species_definitions(*file, count, &error);
    if (!species) {
      if (error.empty()) return 0;
      if (message && message_capacity) std::snprintf(message, message_capacity, "%s", error.c_str());
      return -1;
    }
    for (int i = 0; i < count; ++i) {
      const openmx::io::SpeciesDefinition& s = (*species)[static_cast<size_t>(i)];
      if (!store(s.name, names[i], capacity) || !store(s.basis, basis[i], capacity) ||
          !store(s.pseudopotential, pseudopotentials[i], capacity)) {
        if (message && message_capacity)
          std::snprintf(message, message_capacity, "species %d: a field longer than %zu characters\n", i + 1, capacity - 1);
        return -1;
      }
      if (s.mass_given) mass[i] = s.mass;
    }
    return 1;
  } catch (...) {
    return -1;
  }
}

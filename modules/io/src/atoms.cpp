#include <openmx/io/atoms.h>
#include <openmx/io/atoms.hpp>
#include <openmx/io/species.hpp>   // openmx_io_keyword_files_current

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace openmx::io {

namespace {

// sscanf("%lf") on one token: the longest numeric prefix, failure when none
bool number(const std::string& token, double& value) noexcept
{
  char* end = nullptr;
  value = std::strtod(token.c_str(), &end);
  return end != token.c_str();
}

// sscanf("%i") on one token
bool integer(const std::string& token, int& value) noexcept
{
  char* end = nullptr;
  const long v = std::strtol(token.c_str(), &end, 0);
  if (end == token.c_str()) return false;
  value = static_cast<int>(v);
  return true;
}

}  // namespace

std::optional<std::vector<AtomDefinition>> read_atom_definitions(KeywordFile& file, int expected, bool noncollinear,
                                                                 std::string* error)
{
  static constexpr const char* format_error = "Format error for Atoms.SpeciesAndCoordinates\n";
  if (error) error->clear();
  const std::optional<std::vector<std::string>> lines = file.block("Atoms.SpeciesAndCoordinates");
  if (!lines) {
    if (file.find("<Atoms.SpeciesAndCoordinates") && error) *error = format_error;
    return std::nullopt;
  }
  if (expected < 0 || static_cast<int>(lines->size()) != expected) {
    if (error) *error = format_error;
    return std::nullopt;
  }
  std::vector<AtomDefinition> atoms;
  atoms.reserve(lines->size());
  for (const std::string& line : *lines) {
    AtomDefinition a;
    std::istringstream in(line);
    std::string token;
    // the conversion stops at the first token that does not fit, as sscanf does
    if (in >> token && integer(token, a.index)) {
      a.fields = 1;
      if (in >> a.species) {
        a.fields = 2;
        double* numbers[9] = {&a.position[0], &a.position[1], &a.position[2], &a.spin[0], &a.spin[1],
                              &a.spin_angle[0], &a.spin_angle[1], &a.orbital_angle[0], &a.orbital_angle[1]};
        const int count = noncollinear ? 9 : 5;
        int k = 0;
        for (; k < count && in >> token && number(token, *numbers[k]); ++k) a.fields = 3 + k;
        if (k == count) {
          bool more = true;
          if (noncollinear) {
            more = static_cast<bool>(in >> token) && integer(token, a.constraint);
            if (more) a.fields = 12;
          }
          if (more && in >> a.orbital_polarization) a.fields = noncollinear ? 13 : 8;
        }
      }
    }
    atoms.push_back(std::move(a));
  }
  return atoms;
}

}  // namespace openmx::io

extern "C" int openmx_io_read_atom_definitions(OpenMXKeywordFiles* files, int count, int noncollinear,
                                               OpenMXAtomDefinition* atoms, char* message, size_t message_capacity)
{
  if (message && message_capacity) message[0] = '\0';
  if (!files || count < 0 || (count > 0 && !atoms)) return -1;
  openmx::io::KeywordFile* file = openmx_io_keyword_files_current(files);
  if (!file) return -1;
  try {
    std::string error;
    const auto parsed = openmx::io::read_atom_definitions(*file, count, noncollinear != 0, &error);
    if (!parsed) {
      if (error.empty()) return 0;
      if (message && message_capacity) std::snprintf(message, message_capacity, "%s", error.c_str());
      return -1;
    }
    for (int i = 0; i < count; ++i) {
      const openmx::io::AtomDefinition& a = (*parsed)[static_cast<size_t>(i)];
      OpenMXAtomDefinition& out = atoms[i];
      if (a.species.size() >= sizeof out.species || a.orbital_polarization.size() >= sizeof out.orbital_polarization) {
        if (message && message_capacity)
          std::snprintf(message, message_capacity, "atom %d: a name longer than %zu characters\n", i + 1, sizeof out.species - 1);
        return -1;
      }
      out.fields = a.fields;
      out.index = a.index;
      std::memcpy(out.species, a.species.c_str(), a.species.size() + 1);
      for (int d = 0; d < 3; ++d) out.position[d] = a.position[d];
      for (int s = 0; s < 2; ++s) { out.spin[s] = a.spin[s]; out.spin_angle[s] = a.spin_angle[s]; out.orbital_angle[s] = a.orbital_angle[s]; }
      out.constraint = a.constraint;
      std::memcpy(out.orbital_polarization, a.orbital_polarization.c_str(), a.orbital_polarization.size() + 1);
    }
    return 1;
  } catch (...) {
    return -1;
  }
}

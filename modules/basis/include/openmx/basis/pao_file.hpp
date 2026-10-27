#pragma once

#include <cstddef>
#include <istream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace openmx::basis {

struct PaoMetadata {
  double atom_species = 0;
  double cutoff = 5;
  int mesh_points = 0;
  int lmax = 0;
  int multiplicity = 0;
  int atomic_number() const;
};

enum class PaoReadMode { full, metadata };

// An immutable, independently owned PAO file. Reads neither process-global
// input state nor OpenMX settings. Full reads validate every mesh and value;
// metadata reads do not require or allocate the density/orbital tables.
// Unknown sections (e.g. generator settings/contraction history) are ignored.
// Errors throw with section/line context. No library function exits a process.
class PaoFile {
 public:
  static PaoFile read(std::istream& input, PaoReadMode mode = PaoReadMode::full);
  static PaoFile from_file(const std::string& path, PaoReadMode mode = PaoReadMode::full);
  PaoFile(const PaoFile&) = default;
  PaoFile(PaoFile&&) noexcept = default;
  PaoFile& operator=(const PaoFile& other) {
    if (this != &other) {
      PaoFile replacement(other);
      *this = std::move(replacement);
    }
    return *this;
  }
  PaoFile& operator=(PaoFile&&) noexcept = default;

  const PaoMetadata& metadata() const noexcept { return metadata_; }
  bool has_radial_data() const noexcept { return !radii_.empty(); }
  std::span<const double> log_radii() const noexcept { return log_radii_; }
  std::span<const double> radii() const noexcept { return radii_; }
  // Physical points only, without the application's two padding entries.
  std::span<const double> density() const noexcept { return density_; }
  std::span<const double> orbital(int angular, int multiplicity) const;
  // R_l(r) in the same convention as RadialTable(RadialKind::pao).
  double sample(int angular, int multiplicity, double radius) const;
  // Optional generator eigenvalue in Hartree. Virtual-atom PAOs can omit it.
  std::optional<double> orbital_energy(int angular, int multiplicity) const;

 private:
  PaoFile() = default;
  std::size_t channel_index(int angular, int multiplicity) const;
  PaoMetadata metadata_;
  std::vector<double> log_radii_, radii_, density_;
  std::vector<double> orbitals_; // [l, multiplicity, radial point], contiguous
  std::vector<double> energies_; // empty when the optional Lmax= block is absent
};

} // namespace openmx::basis

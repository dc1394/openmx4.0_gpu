#pragma once

#include <cstddef>
#include <istream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace openmx::basis {

enum class VpsReadMode { full, metadata };
enum class VpsFormat { separable, legacy };

struct VpsMetadata {
  double atom_species = 0, total_electrons = 0, valence_electrons = 0;
  int mesh_points = 0, projector_components = 0, max_angular = 0;
  bool j_dependent = false, partial_core = false;
  VpsFormat format = VpsFormat::separable;
  int spin_channels() const noexcept { return j_dependent ? 2 : 1; }
  int atomic_number() const;
  double core_charge() const noexcept { return valence_electrons+atom_species-total_electrons; }
};

// Immutable raw VPS data. The j-dependent channels remain separate; scalar
// averaging, spin-orbit scaling and charge overrides belong to the consumer.
// Energies/local potentials use Hartree, radii use bohr. Density contains only
// physical points (no application padding). Projectors are stored as written.
// File and intermediate buffers are RAII-owned, with no global parser state.
class VpsFile {
 public:
  static VpsFile read(std::istream& input, VpsReadMode mode = VpsReadMode::full);
  static VpsFile from_file(const std::string& path, VpsReadMode mode = VpsReadMode::full);
  VpsFile(const VpsFile&) = default;
  VpsFile(VpsFile&&) noexcept = default;
  VpsFile& operator=(const VpsFile& other) {
    if (this != &other) { VpsFile replacement(other); *this = std::move(replacement); }
    return *this;
  }
  VpsFile& operator=(VpsFile&&) noexcept = default;

  const VpsMetadata& metadata() const noexcept { return metadata_; }
  bool has_radial_data() const noexcept { return !radii_.empty(); }
  std::span<const int> angular() const noexcept { return angular_; }
  std::span<const double> log_radii() const noexcept { return log_radii_; }
  std::span<const double> radii() const noexcept { return radii_; }
  std::span<const double> local_potential() const noexcept { return local_; }
  // Empty when charge.pcc.calc is off, including metadata-only reads.
  std::span<const double> partial_core_density() const noexcept { return partial_core_; }
  // Contiguous borrowed storage in the order documented below.
  std::span<const double> projector_values() const noexcept { return projectors_; }
  std::span<const double> projector_energies() const noexcept { return energies_; }
  std::span<const double> projector(std::size_t index, int spin_channel) const;
  double sample_projector(std::size_t index, int spin_channel, double radius) const;
  // Legacy metadata-only reads omit the projection.energies table.
  double projector_energy(std::size_t index, int spin_channel) const;

 private:
  VpsFile() = default;
  std::size_t channel_index(std::size_t index, int spin_channel) const;
  VpsMetadata metadata_;
  std::vector<int> angular_;
  std::vector<double> energies_; // [projector, spin channel]
  std::vector<double> log_radii_, radii_, local_, partial_core_;
  std::vector<double> projectors_; // [projector, spin channel, radial point]
};

} // namespace openmx::basis

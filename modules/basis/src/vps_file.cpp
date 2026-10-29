#include <openmx/basis/vps_file.hpp>
#include <openmx/basis/radial_table.hpp>
#include "text_document.hpp"

#include <climits>
#include <fstream>

namespace openmx::basis {
namespace {
void append_angular(std::vector<int>& angular, VpsMetadata& metadata, int l)
{
  if (l < 0 || l > 29 || metadata.projector_components > INT_MAX-2-(2*l+1))
    throw std::invalid_argument("VPS invalid projector angular momentum or count");
  angular.push_back(l);
  metadata.projector_components += 2*l+1;
  if (l > metadata.max_angular) metadata.max_angular = l;
}
void check_count(int count, std::size_t input_size)
{
  if (count < 0 || count > INT_MAX-2 || static_cast<std::size_t>(count) > input_size/3)
    throw std::invalid_argument("VPS invalid projector count");
}
} // namespace

int VpsMetadata::atomic_number() const
{
  if (!std::isfinite(atom_species) || atom_species < 0 || atom_species > INT_MAX)
    throw std::invalid_argument("invalid VPS atomic species");
  return atom_species > 1e-15 && atom_species < 1 ? 1 : static_cast<int>(atom_species);
}

VpsFile VpsFile::read(std::istream& input, VpsReadMode mode)
{
  if (mode != VpsReadMode::full && mode != VpsReadMode::metadata)
    throw std::invalid_argument("invalid VPS read mode");
  const detail::Document document(input,"VPS");
  VpsFile result;
  auto& m = result.metadata_;
  m.atom_species = document.scalar<double>("AtomSpecies");
  (void)m.atomic_number();
  m.total_electrons = document.scalar<double>("total.electron",m.atom_species);
  m.valence_electrons = document.scalar<double>("valence.electron",0.);
  m.mesh_points = document.scalar<int>("grid.num.output");
  m.j_dependent = document.logical("j.dependent.pseudo.potentials");
  m.partial_core = document.logical("charge.pcc.calc");
  if (m.mesh_points < 6 || m.mesh_points > INT_MAX-2 || !std::isfinite(m.core_charge()))
    throw std::invalid_argument("VPS invalid mesh size or charge");

  if (document.find("<project.energies")) {
    auto energies = document.section("project.energies");
    const int count = energies.number<int>();
    check_count(count,document.size());
    result.angular_.reserve(count);
    result.energies_.reserve(detail::checked_product(count,m.spin_channels()));
    for (int i = 0; i < count; ++i) {
      append_angular(result.angular_,m,energies.number<int>());
      for (int spin = 0; spin < m.spin_channels(); ++spin)
        result.energies_.push_back(energies.number<double>());
    }
    energies.end();
  } else {
    m.format = VpsFormat::legacy;
    // The historical format has a single scalar channel, even if unrelated
    // generator input mentions spin-orbit settings.
    if (m.j_dependent) throw std::invalid_argument("legacy VPS format cannot carry j-dependent projectors");
    const int count = document.scalar<int>("number.vps");
    const int local = document.scalar<int>("local.part.vps");
    check_count(count,document.size());
    if (!count) throw std::invalid_argument("VPS has no legacy channel descriptions");
    auto channels = document.section("pseudo.NandL");
    // Preserve the historical row-index rule: an out-of-range local label
    // excludes no row (some generators use a separately built local part).
    result.angular_.reserve(count);
    for (int i = 0; i < count; ++i) {
      (void)channels.number<int>(); // legacy row label, not a storage offset
      (void)channels.number<int>(); // principal quantum number
      const int l = channels.number<int>();
      (void)channels.number<double>(); // generator cutoff
      if (l < 0 || l > 29) channels.fail("invalid angular momentum");
      if (i != local) append_angular(result.angular_,m,l);
    }
    channels.end();
    if (mode == VpsReadMode::full && !result.angular_.empty()) {
      auto energies = document.section("projection.energies");
      result.energies_.reserve(result.angular_.size());
      for (std::size_t i = 0; i < result.angular_.size(); ++i) {
        (void)energies.number<int>(); // legacy row label
        result.energies_.push_back(energies.number<double>());
      }
      energies.end();
    }
  }
  if (mode == VpsReadMode::metadata) return result;
  const auto rows = detail::checked_product(result.angular_.size(),m.spin_channels());
  const auto count = detail::checked_product(rows,m.mesh_points);
  if (count > document.size() || static_cast<std::size_t>(m.mesh_points) > document.size()/3)
    throw std::invalid_argument("VPS radial dimensions exceed the input size");
  result.log_radii_.resize(m.mesh_points);
  result.radii_.resize(m.mesh_points);
  result.local_.resize(m.mesh_points);
  result.projectors_.resize(count);
  auto potentials = document.section("Pseudo.Potentials");
  for (std::size_t i = 0; i < result.radii_.size(); ++i) {
    detail::read_mesh_point(potentials,i,result.log_radii_,result.radii_);
    result.local_[i] = potentials.number<double>();
    for (std::size_t row = 0; row < rows; ++row)
      result.projectors_[row*result.radii_.size()+i] = potentials.number<double>();
  }
  potentials.end();
  if (m.partial_core) {
    result.partial_core_.resize(m.mesh_points);
    auto density = document.section("density.PCC");
    for (std::size_t i = 0; i < result.radii_.size(); ++i) {
      const double logarithm = density.number<double>(), radius = density.number<double>();
      if (std::abs(logarithm-result.log_radii_[i]) > 1e-10*(1+std::abs(result.log_radii_[i])) ||
          std::abs(radius-result.radii_[i]) > 1e-10*(1+std::abs(result.radii_[i])))
        density.fail("potential and partial-core meshes disagree");
      result.partial_core_[i] = density.number<double>();
    }
    density.end();
  }
  return result;
}

VpsFile VpsFile::from_file(const std::string& path, VpsReadMode mode)
{
  std::ifstream input(path,std::ios::binary);
  if (!input) throw std::runtime_error("cannot open VPS file: "+path);
  try { return read(input,mode); }
  catch (const std::exception& error) { throw std::runtime_error(path+": "+error.what()); }
}

std::size_t VpsFile::channel_index(std::size_t index, int spin_channel) const
{
  if (index >= angular_.size() || spin_channel < 0 || spin_channel >= metadata_.spin_channels())
    throw std::out_of_range("VPS projector channel index");
  return index*metadata_.spin_channels()+spin_channel;
}

std::span<const double> VpsFile::projector(std::size_t index, int spin_channel) const
{
  const auto row = channel_index(index,spin_channel);
  if (!has_radial_data()) throw std::logic_error("VPS metadata-only file has no radial projectors");
  return {projectors_.data()+row*radii_.size(),radii_.size()};
}

double VpsFile::projector_energy(std::size_t index, int spin_channel) const
{
  const auto channel = channel_index(index,spin_channel);
  if (energies_.empty()) throw std::logic_error("legacy VPS metadata has no projector energies");
  return energies_[channel];
}

double VpsFile::sample_projector(std::size_t index, int spin_channel, double radius) const
{
  const auto values = projector(index,spin_channel);
  return sample_radial_row(RadialKind::projector,angular_[index],radii_,values,0,radius);
}

} // namespace openmx::basis

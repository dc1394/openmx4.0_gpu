#include <openmx/basis/pao_file.hpp>
#include <openmx/basis/radial_table.hpp>
#include "text_document.hpp"

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace openmx::basis {
using detail::Document;
using detail::equal;
using detail::checked_product;
using detail::read_mesh_point;

int PaoMetadata::atomic_number() const
{
  if (!std::isfinite(atom_species) || atom_species < 0 || atom_species > INT_MAX)
    throw std::invalid_argument("invalid PAO atomic species");
  return atom_species > 1e-15 && atom_species < 1 ? 1 : static_cast<int>(atom_species);
}

PaoFile PaoFile::read(std::istream& input, PaoReadMode mode)
{
  if (mode != PaoReadMode::full && mode != PaoReadMode::metadata)
    throw std::invalid_argument("invalid PAO read mode");
  const Document document(input,"PAO");
  PaoFile result;
  auto& m = result.metadata_;
  m.atom_species = document.scalar<double>("AtomSpecies");
  m.mesh_points = document.scalar<int>("grid.num.output");
  m.cutoff = document.scalar<double>("radial.cutoff.pao",5.);
  m.lmax = document.scalar<int>("PAO.Lmax");
  m.multiplicity = document.scalar<int>("PAO.Mul");
  if (m.atom_species < 0 || m.atom_species > INT_MAX || m.mesh_points < 3 ||
      m.mesh_points > INT_MAX-2 || m.lmax < 0 || m.lmax > 29 || m.multiplicity < 1 || m.cutoff <= 0)
    throw std::invalid_argument("PAO invalid species, dimensions or cutoff");
  const auto channels = checked_product(static_cast<std::size_t>(m.lmax+1),m.multiplicity);
  if (const auto line = document.find("Lmax=")) {
    if (channels > document.size()/5)
      throw std::invalid_argument("PAO energy dimensions exceed the input size");
    // Preserve the data set's labels, without retaining any input stream.
    auto tokens = document.after(*line,"orbital energies");
    result.energies_.assign(channels,std::numeric_limits<double>::quiet_NaN());
    for (std::size_t channel = 0; channel < channels; ++channel) {
      if (!equal(tokens.next(),"l") || !equal(tokens.next(),"mu")) tokens.fail("expected l mu energy labels");
      const int l = tokens.number<int>(), mu = tokens.number<int>();
      if (l < 0 || l > m.lmax || mu < 0 || mu >= m.multiplicity) tokens.fail("invalid energy channel");
      auto& energy = result.energies_[static_cast<std::size_t>(l)*m.multiplicity+mu];
      if (std::isfinite(energy)) tokens.fail("duplicate energy channel");
      energy = tokens.number<double>();
    }
  }
  if (mode == PaoReadMode::metadata) return result;

  const auto orbital_size = checked_product(channels,m.mesh_points);
  if (orbital_size > document.size() || static_cast<std::size_t>(m.mesh_points) > document.size()/3)
    throw std::invalid_argument("PAO radial dimensions exceed the input size");

  result.log_radii_.resize(m.mesh_points);
  result.radii_.resize(m.mesh_points);
  result.density_.resize(m.mesh_points);
  auto density = document.section("valence.charge.density");
  for (std::size_t i = 0; i < result.radii_.size(); ++i) {
    read_mesh_point(density,i,result.log_radii_,result.radii_);
    result.density_[i] = density.number<double>();
  }
  density.end();
  result.orbitals_.resize(orbital_size);
  std::vector<double> log_mesh(m.mesh_points), mesh(m.mesh_points);
  for (int l = 0; l <= m.lmax; ++l) {
    auto table = document.section("pseudo.atomic.orbitals.L="+std::to_string(l));
    for (std::size_t i = 0; i < mesh.size(); ++i) {
      read_mesh_point(table,i,log_mesh,mesh);
      // The file may print the same mesh at different decimal precisions.
      if (std::abs(mesh[i]-result.radii_[i]) > 1e-10*(1+std::abs(result.radii_[i])) ||
          std::abs(log_mesh[i]-result.log_radii_[i]) > 1e-10*(1+std::abs(result.log_radii_[i])))
        table.fail("density and orbital meshes disagree");
      for (int mu = 0; mu < m.multiplicity; ++mu)
        result.orbitals_[(static_cast<std::size_t>(l)*m.multiplicity+mu)*mesh.size()+i] = table.number<double>();
    }
    table.end();
    // Legacy Read_PAO uses the coordinates of the last orbital block.
    result.log_radii_ = log_mesh;
    result.radii_ = mesh;
  }
  return result;
}

PaoFile PaoFile::from_file(const std::string& path, PaoReadMode mode)
{
  std::ifstream input(path,std::ios::binary);
  if (!input) throw std::runtime_error("cannot open PAO file: "+path);
  try { return read(input,mode); }
  catch (const std::exception& error) { throw std::runtime_error(path+": "+error.what()); }
}

std::size_t PaoFile::channel_index(int angular, int multiplicity) const
{
  if (angular < 0 || angular > metadata_.lmax || multiplicity < 0 || multiplicity >= metadata_.multiplicity)
    throw std::out_of_range("PAO channel index");
  return static_cast<std::size_t>(angular)*metadata_.multiplicity+multiplicity;
}

std::span<const double> PaoFile::orbital(int angular, int multiplicity) const
{
  const auto channel = channel_index(angular,multiplicity);
  if (!has_radial_data()) throw std::logic_error("PAO metadata-only file has no orbitals");
  return {orbitals_.data()+channel*radii_.size(),radii_.size()};
}

double PaoFile::sample(int angular, int multiplicity, double radius) const
{
  return sample_radial_row(RadialKind::pao,angular,radii_,orbital(angular,multiplicity),0,radius);
}

std::optional<double> PaoFile::orbital_energy(int angular, int multiplicity) const
{
  const auto channel = channel_index(angular,multiplicity);
  if (energies_.empty()) return std::nullopt;
  return energies_[channel];
}

} // namespace openmx::basis

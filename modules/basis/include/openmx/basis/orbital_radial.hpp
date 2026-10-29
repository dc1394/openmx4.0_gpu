#pragma once
#include <openmx/basis/orbital_radial.h>
#include <openmx/basis/radial_table.hpp>
#include <span>
#include <utility>
#include <vector>

namespace openmx::basis {
using OrbitalRadialSample=::OpenMXOrbitalRadialSample;

// Checked, borrowed real-space orbital rows. Input grid, descriptors and values
// must remain valid and unchanged during evaluation. No runtime/global state.
// In contrast to RadialKind::pao, the first interval is not clamped to interval
// 2 and the origin uses the native orbital cubic continuation. See the C
// header for units and endpoint conventions. Sampling allocates no memory.
class OrbitalRadialTableView {
 public:
  OrbitalRadialTableView(std::span<const double> radii,std::span<const RadialChannel> channels);
  std::size_t channel_count() const noexcept { return channels_.size(); }
  int angular(std::size_t channel) const;
  OrbitalRadialSample sample(std::size_t channel,double radius) const;
  // One shared stencil for all rows. Rejects invalid arguments before writing;
  // only the first channel_count entries are changed. Safe for concurrent
  // readers with separate output buffers.
  void evaluate(double radius,std::span<OrbitalRadialSample> out) const;
 private:
  friend class OrbitalRadialTable;
  std::span<const double> radii_;
  std::span<const RadialChannel> channels_;
};

// Immutable, independently owned grid/rows. Copies own separate storage;
// moves transfer it. Moved-from owners reject sampling until reassigned.
class OrbitalRadialTable {
 public:
  OrbitalRadialTable(std::span<const double> radii,std::span<const RadialChannel> channels);
  OrbitalRadialTable(const OrbitalRadialTable&) = default;
  OrbitalRadialTable(OrbitalRadialTable&&) noexcept = default;
  OrbitalRadialTable& operator=(const OrbitalRadialTable& other) {
    if(this!=&other){OrbitalRadialTable replacement(other);*this=std::move(replacement);}return *this;
  }
  OrbitalRadialTable& operator=(OrbitalRadialTable&&) noexcept = default;
  std::size_t channel_count() const noexcept { return angular_.size(); }
  int angular(std::size_t channel) const { return angular_.at(channel); }
  OrbitalRadialSample sample(std::size_t channel,double radius) const;
  void evaluate(double radius,std::span<OrbitalRadialSample> out) const;
 private:
  explicit OrbitalRadialTable(const OrbitalRadialTableView& validated);
  std::vector<double> radii_,values_;
  std::vector<int> angular_;
};
} // namespace openmx::basis

#pragma once

#include <openmx/core/radial.hpp>

namespace openmx::core {

enum class RadialKind { pao, projector, neutral_potential };

// A borrowed input row, copied by RadialTable before numerical work begins.
struct RadialChannel {
  int angular;
  std::span<const double> values;
};

// Borrowed scalar adapter for callers whose table still belongs to the C
// application. Shape/coordinate checks are O(1); the caller must provide a
// strictly increasing, finite grid and finite values (RadialTable checks these
// once on construction). No storage is retained or allocated by this function.
double sample_radial_row(RadialKind kind, int angular, std::span<const double> radii,
    std::span<const double> values, double cutoff, double radius);

class RadialTable {
 public:
  RadialTable(RadialKind kind, std::span<const double> radii,
              std::span<const RadialChannel> channels, double cutoff = 0);
  RadialTable(const RadialTable&) = default;
  RadialTable(RadialTable&&) noexcept = default;
  RadialTable& operator=(const RadialTable& other) {
    if (this != &other) {
      RadialTable replacement(other);
      *this = std::move(replacement);
    }
    return *this;
  }
  RadialTable& operator=(RadialTable&&) noexcept = default;
  std::size_t channel_count() const noexcept { return angular_.size(); }
  int angular(std::size_t channel) const { return angular_.at(channel); }
  double sample(std::size_t channel, double radius) const;

 private:
  RadialKind kind_;
  double cutoff_;
  std::vector<double> radii_;
  std::vector<int> angular_;
  Tensor<double,2> values_; // [channel, tabulated radius], independently owned
};

RadialSamples sample_table(const RadialTable& table, RadialQuadrature grid);

} // namespace openmx::core

#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace openmx::basis {

enum class RadialKind { pao, projector, neutral_potential, neutral_projector };
enum class ReciprocalKind { pao, projector };

// Uniform grid k[i]=i*step, computed without a global coordinate array.
// Borrowed finite values; no allocations. Preserves the legacy first-cell
// extrapolation and the projector's small-spacing linear fallback.
double sample_reciprocal_row(ReciprocalKind kind, std::span<const double> values,
    double step, double wavevector);

// A borrowed input row; RadialTable copies it, RadialTableView borrows it.
struct RadialChannel {
  int angular;
  std::span<const double> values;
};

// Borrowed scalar adapter for legacy C consumers. Shape/coordinate checks
// are O(1); the caller must provide a strictly increasing, finite grid and
// finite values (RadialTable and RadialTableView check these on construction).
// No storage is retained or allocated by this function.
double sample_radial_row(RadialKind kind, int angular, std::span<const double> radii,
    std::span<const double> values, double cutoff, double radius);

// Checks all coordinates/channels/values once, without copying numeric data.
// The mesh, channel descriptors and rows must outlive this view and remain
// valid and unchanged during sampling. No process-global state is retained.
class RadialTableView {
 public:
  RadialTableView(RadialKind kind, std::span<const double> radii,
                 std::span<const RadialChannel> channels, double cutoff = 0);
  std::size_t channel_count() const noexcept { return channels_.size(); }
  int angular(std::size_t channel) const;
  double sample(std::size_t channel, double radius) const;

 private:
  friend class RadialTable;
  const RadialChannel& channel(std::size_t index) const;
  RadialKind kind_;
  double cutoff_;
  std::span<const double> radii_;
  std::span<const RadialChannel> channels_;
};

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
  explicit RadialTable(const RadialTableView& validated);
  RadialKind kind_;
  double cutoff_;
  std::vector<double> radii_;
  std::vector<int> angular_;
  std::vector<double> values_; // [channel * radii_.size() + point], independently owned
};

} // namespace openmx::basis

#include <openmx/core/radial_table.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace openmx::core {
namespace {

struct Interpolated { double value, derivative; };

template<class Grid>
Interpolated interpolate(const Grid& radii, const double* values,
                         std::size_t m, double radius, bool derivative = false,
                         double minimum_spacing = -1)
{
  const double h2 = radii[m]-radii[m-1];
  const double f2 = values[m-1], f3 = values[m];
  double h1, h3, f1, f4;
  // Select the edge stencil before reading its entries. The original VNAF
  // read past the upper grid before replacing that value with the ghost point.
  if (m == 1) {
    h3 = radii[m+1]-radii[m]; f4 = values[m+1];
    h1 = -(h2+h3); f1 = f4;
  } else if (m == radii.size()-1) {
    h1 = radii[m-1]-radii[m-2]; f1 = values[m-2];
    h3 = -(h1+h2); f4 = f1;
  } else {
    h1 = radii[m-1]-radii[m-2]; h3 = radii[m+1]-radii[m];
    f1 = values[m-2]; f4 = values[m+1];
  }
  if (minimum_spacing >= 0 &&
      (std::abs(h1) <= minimum_spacing || std::abs(h2) <= minimum_spacing ||
       std::abs(h3) <= minimum_spacing || std::abs(h1+h2) <= minimum_spacing ||
       std::abs(h2+h3) <= minimum_spacing)) {
    if (std::abs(h2) <= minimum_spacing) return {f2,0};
    return {f2+(f3-f2)*(radius-radii[m-1])/h2,(f3-f2)/h2};
  }
  const double g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
  const double g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);
  const double y1 = (radius-radii[m-1])/h2;
  const double y2 = (radius-radii[m])/h2;
  const double y12 = y1*y1, y22 = y2*y2;
  const double value = y22*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)
                     + y12*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1);
  double slope = 0;
  if (derivative)
    slope = 2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)
          + y22*(2.0*f2+h2*g1)/h2
          + 2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)
          - y12*(2.0*f3-h2*g2)/h2;
  return {value,slope};
}

} // namespace

double sample_reciprocal_row(ReciprocalKind kind, std::span<const double> values,
    double step, double wavevector)
{
  if ((kind != ReciprocalKind::pao && kind != ReciprocalKind::projector) ||
      values.empty() || values.size() > INT_MAX || !std::isfinite(step) || step <= 0 ||
      !std::isfinite((values.size()-1)*step) || !std::isfinite(wavevector))
    throw std::invalid_argument("invalid reciprocal interpolation row or coordinate");
  struct UniformGrid {
    std::size_t count;
    double step;
    double operator[](std::size_t i) const { return static_cast<double>(i)*step; }
    std::size_t size() const { return count; }
  } grid{values.size(),step};
  if (wavevector > grid[grid.size()-1]) return 0;
  if (values.size() == 1) return values[0];
  if (values.size() == 2) {
    if (wavevector <= 0 || step <= 1e-14) return values[0];
    return values[0]+(values[1]-values[0])*wavevector/step;
  }
  std::size_t m = 1;
  if (wavevector >= 0) {
    std::size_t low = 0, high = grid.size()-1;
    while (high-low > 1) {
      const auto middle = (low+high)/2;
      if (grid[middle] < wavevector) low = middle;
      else high = middle;
    }
    m = std::max(std::size_t{2},high);
  }
  return interpolate(grid,values.data(),m,wavevector,false,
      kind == ReciprocalKind::projector ? 1e-14 : -1).value;
}

RadialTableView::RadialTableView(RadialKind kind, std::span<const double> radii,
    std::span<const RadialChannel> channels, double cutoff)
  : kind_(kind), cutoff_(cutoff), radii_(radii), channels_(channels)
{
  if (kind != RadialKind::pao && kind != RadialKind::projector &&
      kind != RadialKind::neutral_potential)
    throw std::invalid_argument("unknown radial table kind");
  if (radii.size() < (kind == RadialKind::neutral_potential ? 6u : 3u) ||
      radii.size() > INT_MAX || channels.size() > INT_MAX)
    throw std::invalid_argument("invalid radial table dimensions");
  for (std::size_t i = 0; i < radii.size(); ++i)
    if (!std::isfinite(radii[i]) || radii[i] < 0 || (i && radii[i] <= radii[i-1]))
      throw std::invalid_argument("radial table grid must be finite and strictly increasing");
  if (kind == RadialKind::neutral_potential &&
      (channels.size() != 1 || !std::isfinite(cutoff) || cutoff < 0 || cutoff > radii.back()))
    throw std::invalid_argument("invalid neutral-potential cutoff or channel count");
  for (const auto& channel : channels) {
    if (channel.angular < 0 || channel.angular > 29 || channel.values.size() != radii.size() ||
        (kind == RadialKind::neutral_potential && channel.angular != 0))
      throw std::invalid_argument("invalid radial table channel");
    for (double value : channel.values)
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite radial table value");
  }
}

const RadialChannel& RadialTableView::channel(std::size_t index) const
{
  if (index >= channels_.size()) throw std::out_of_range("radial channel index");
  return channels_[index];
}

int RadialTableView::angular(std::size_t index) const { return channel(index).angular; }

double RadialTableView::sample(std::size_t index, double radius) const
{
  const auto& row = channel(index);
  return sample_radial_row(kind_,row.angular,radii_,row.values,cutoff_,radius);
}

RadialTable::RadialTable(RadialKind kind, std::span<const double> radii,
    std::span<const RadialChannel> channels, double cutoff)
  : RadialTable(RadialTableView(kind,radii,channels,cutoff)) {}

RadialTable::RadialTable(const RadialTableView& validated)
  : kind_(validated.kind_), cutoff_(validated.cutoff_),
    radii_(validated.radii_.begin(),validated.radii_.end()),
    values_({validated.channel_count(),validated.radii_.size()})
{
  angular_.reserve(validated.channel_count());
  auto data = values_.view();
  for (std::size_t row = 0; row < validated.channel_count(); ++row) {
    const auto& channel = validated.channels_[row];
    angular_.push_back(channel.angular);
    for (std::size_t i = 0; i < radii_.size(); ++i)
      element(data,row,i) = channel.values[i];
  }
}

double RadialTable::sample(std::size_t channel, double radius) const
{
  const int l = angular_.at(channel);
  return sample_radial_row(kind_,l,radii_,
      {values_.data()+channel*radii_.size(),radii_.size()},cutoff_,radius);
}

double sample_radial_row(RadialKind kind, int l, std::span<const double> radii,
    std::span<const double> row, double cutoff, double radius)
{
  if ((kind != RadialKind::pao && kind != RadialKind::projector && kind != RadialKind::neutral_potential) ||
      radii.size() < (kind == RadialKind::neutral_potential ? 6u : 3u) ||
      row.size() != radii.size() || l < 0 || l > 29 ||
      (kind == RadialKind::neutral_potential &&
       (l != 0 || !std::isfinite(cutoff) || cutoff < 0 || cutoff > radii.back())))
    throw std::invalid_argument("invalid radial interpolation row");
  if (!std::isfinite(radius) || radius < 0)
    throw std::invalid_argument("invalid radial sampling coordinate");
  const double* values = row.data();
  if (kind == RadialKind::neutral_potential && radius > cutoff) return 0;
  if (radius > radii.back()) return 0;
  if (radius < radii.front()) {
    if (kind == RadialKind::pao) return l == 0 ? values[0] : 0;
    if (kind == RadialKind::projector)
      return l == 0 ? interpolate(radii,values,1,radius).value : 0;
    const double rm = radii[4];
    const auto at_rm = interpolate(radii,values,4,rm,true);
    const double a = 0.5*at_rm.derivative/rm;
    const double b = at_rm.value-a*rm*rm;
    return a*radius*radius+b;
  }
  // lower_bound, rather than upper_bound, retains the original stencil at
  // an exactly tabulated point, including the first and last grid points.
  std::size_t m = std::lower_bound(radii.begin()+1,radii.end(),radius)-radii.begin();
  if (kind != RadialKind::projector && m < 2) m = 2;
  return interpolate(radii,values,m,radius).value;
}

namespace {
template<class Table>
RadialSamples sample_rows(const Table& table, RadialQuadrature grid)
{
  std::vector<Orbital> orbitals;
  orbitals.reserve(table.channel_count());
  Tensor<double,2> values({table.channel_count(),grid.radii.size()});
  auto data = values.view();
  for (std::size_t channel = 0; channel < table.channel_count(); ++channel) {
    orbitals.push_back({table.angular(channel),static_cast<int>(channel)});
    for (std::size_t i = 0; i < grid.radii.size(); ++i)
      element(data,channel,i) = table.sample(channel,grid.radii[i]);
  }
  return {std::move(grid),std::move(orbitals),std::move(values)};
}
} // namespace

RadialSamples sample_table(const RadialTable& table, RadialQuadrature grid)
{
  return sample_rows(table,std::move(grid));
}

RadialSamples sample_table(const RadialTableView& table, RadialQuadrature grid)
{
  return sample_rows(table,std::move(grid));
}

} // namespace openmx::core

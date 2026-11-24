#include <openmx/exx/two_center.hpp>

#include <openmx/basis/bessel.hpp>
#include <openmx/basis/gaunt.hpp>
#include <openmx/basis/harmonics.hpp>
#include <openmx/basis/real_harmonics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace openmx::exx {
namespace {

constexpr int maximum_bessel_order = 29;
// basis::gaunt() accepts triangles whose degrees sum to 38 at most; the
// largest sum of a table is 2*(maximum_left + maximum_right).
constexpr int maximum_angular_sum = 19;

void check_values(const RadialGrid& grid, std::span<const double> values)
{
  if (values.size() != grid.size())
    throw std::invalid_argument("a radial function does not match its grid");
}

void check_angular(int angular)
{
  if (angular < 0 || angular > maximum_bessel_order)
    throw std::invalid_argument("the angular momentum of a radial function must be in [0,29]");
}

bool odd(int value) { return value%2 != 0; }

} // namespace

void check(const RadialGrid& grid)
{
  if (!std::isfinite(grid.first) || grid.first < 0.0 || !std::isfinite(grid.step) || grid.step <= 0.0 ||
      grid.intervals < 3 || !std::isfinite(grid.last()))
    throw std::invalid_argument("invalid radial grid");
}

double radial_integral(const RadialGrid& grid, std::span<const double> a, std::span<const double> b)
{
  check(grid);
  check_values(grid,a);
  check_values(grid,b);
  const std::size_t last = grid.size()-1;
  double sum = 0.0;
  for (std::size_t i = 0; i <= last; ++i) {
    const double r = grid.radius(i);
    double term = (r*r)*a[i]*b[i];
    if (i == 0 || i == last) term *= 0.5;
    sum += term;
  }
  return sum*grid.step;
}

double radial_moment(const RadialGrid& grid, std::span<const double> a, int power)
{
  check(grid);
  check_values(grid,a);
  if (power < 0 || power > maximum_bessel_order)
    throw std::invalid_argument("the power of a radial moment must be in [0,29]");
  const std::size_t last = grid.size()-1;
  double sum = 0.0;
  for (std::size_t i = 0; i <= last; ++i) {
    double term = std::pow(grid.radius(i),power+2)*a[i];
    if (i == 0 || i == last) term *= 0.5;
    sum += term;
  }
  return sum*grid.step;
}

ReciprocalGrid::ReciprocalGrid(std::vector<double> nodes, std::vector<double> weights)
  : nodes_(std::move(nodes)), weights_(std::move(weights))
{
  if (nodes_.empty() || nodes_.size() != weights_.size())
    throw std::invalid_argument("a reciprocal grid needs as many weights as nodes");
  for (std::size_t i = 0; i < nodes_.size(); ++i)
    if (!std::isfinite(nodes_[i]) || nodes_[i] < 0.0 || (i && nodes_[i] <= nodes_[i-1]) ||
        !std::isfinite(weights_[i]) || weights_[i] <= 0.0)
      throw std::invalid_argument("reciprocal nodes must increase from a nonnegative value, with positive weights");
}

ReciprocalGrid ReciprocalGrid::uniform(double maximum, int intervals, double minimum)
{
  if (!std::isfinite(maximum) || !std::isfinite(minimum) || minimum < 0.0 || maximum <= minimum || intervals < 1)
    throw std::invalid_argument("invalid uniform reciprocal grid");
  const double step = (maximum-minimum)/intervals;
  std::vector<double> nodes(static_cast<std::size_t>(intervals)+1), weights(nodes.size(),step);
  for (std::size_t i = 0; i < nodes.size(); ++i) nodes[i] = minimum+static_cast<double>(i)*step;
  weights.front() = weights.back() = 0.5*step;
  return ReciprocalGrid(std::move(nodes),std::move(weights));
}

std::vector<ReciprocalFunction> bessel_transform(const RadialGrid& grid,
    std::span<const RadialFunction> functions, const ReciprocalGrid& reciprocal)
{
  check(grid);
  int maximum = 0;
  for (const RadialFunction& f : functions) {
    check_angular(f.angular);
    check_values(grid,f.values);
    maximum = std::max(maximum,f.angular);
  }
  const std::size_t n = grid.size(), nodes = reciprocal.size();
  std::vector<ReciprocalFunction> result(functions.size());
  for (std::size_t f = 0; f < functions.size(); ++f)
    result[f] = {functions[f].angular,std::vector<double>(nodes)};
  if (functions.empty()) return result;

  // table[l*n + i] = c_i r_i^2 j_l(kappa r_i), with the trapezoid factor c_i.
  std::vector<double> table(static_cast<std::size_t>(maximum+1)*n);
  for (std::size_t k = 0; k < nodes; ++k) {
    const double kappa = reciprocal.nodes()[k];
    for (std::size_t i = 0; i < n; ++i) {
      const double r = grid.radius(i);
      const basis::SphericalBesselSample sample(kappa*r,maximum);
      const std::span<const double> values = sample.values();
      for (int l = 0; l <= maximum; ++l) {
        double weighted = values[static_cast<std::size_t>(l)]*(r*r);
        if (i == 0 || i == n-1) weighted *= 0.5;
        table[static_cast<std::size_t>(l)*n+i] = weighted;
      }
    }
    for (std::size_t f = 0; f < functions.size(); ++f) {
      const double* row = table.data()+static_cast<std::size_t>(functions[f].angular)*n;
      const std::vector<double>& input = functions[f].values;
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) sum += row[i]*input[i];
      result[f].values[k] = sum*grid.step;
    }
  }
  return result;
}

ReciprocalFunction bessel_transform(const RadialGrid& grid, const RadialFunction& function,
    const ReciprocalGrid& reciprocal)
{
  return std::move(bessel_transform(grid,std::span<const RadialFunction>(&function,1),reciprocal).front());
}

RadialFunction coulomb_potential(const RadialGrid& grid, const RadialFunction& function)
{
  check(grid);
  check_angular(function.angular);
  check_values(grid,function.values);
  const int l = function.angular;
  const std::size_t n = grid.size();
  const double h = grid.step;
  const std::vector<double>& f = function.values;

  // Integrands of the inner and of the outer integral.  At r = 0 the outer
  // integrand f r^(1-l) takes its limit for a function that behaves as r^l.
  std::vector<double> inner(n), outer(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double r = grid.radius(i);
    inner[i] = f[i]*std::pow(r,l+2);
    if (r > 0.0) outer[i] = f[i]*std::pow(r,1-l);
    else outer[i] = l == 1 ? f[i] : 0.0;
  }
  // int_{r_i}^{r_{i+1}} y dr, exact for cubic polynomials.
  const auto interval = [n,h](const std::vector<double>& y, std::size_t i) {
    if (i == 0) return h/24.0*(9.0*y[0]+19.0*y[1]-5.0*y[2]+y[3]);
    if (i == n-2) return h/24.0*(y[n-4]-5.0*y[n-3]+19.0*y[n-2]+9.0*y[n-1]);
    return h/24.0*(-y[i-1]+13.0*y[i]+13.0*y[i+1]-y[i+2]);
  };
  std::vector<double> below(n), above(n);
  // Between the origin and the first node: f(r) = f(first) (r/first)^l.
  below[0] = grid.first > 0.0 ? f[0]*std::pow(grid.first,l+3)/(2.0*l+3.0) : 0.0;
  for (std::size_t i = 0; i+1 < n; ++i) below[i+1] = below[i]+interval(inner,i);
  above[n-1] = 0.0;
  for (std::size_t i = n-1; i-- > 0;) above[i] = above[i+1]+interval(outer,i);

  RadialFunction result{l,std::vector<double>(n)};
  const double factor = 4.0*std::numbers::pi/(2.0*l+1.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double r = grid.radius(i);
    if (r > 0.0) result.values[i] = factor*(below[i]*std::pow(r,-l-1)+std::pow(r,l)*above[i]);
    else result.values[i] = l == 0 ? factor*above[i] : 0.0;
  }
  return result;
}

RealHarmonics::RealHarmonics(int maximum_degree) : maximum_(maximum_degree)
{
  if (maximum_degree < 0 || maximum_degree > maximum_bessel_order)
    throw std::invalid_argument("the degree of the real harmonics must be in [0,29]");
  const basis::RealHarmonicTransform transform(maximum_degree);
  const std::size_t width = 2*static_cast<std::size_t>(maximum_degree)+1;
  transform_.assign((static_cast<std::size_t>(maximum_degree)+1)*width*width,{0.0,0.0});
  for (int l = 0; l <= maximum_degree; ++l)
    for (int row = 0; row <= 2*l; ++row)
      for (int column = 0; column <= 2*l; ++column) {
        const dcomplex c = transform.coefficient(l,row,column);
        transform_[(static_cast<std::size_t>(l)*width+static_cast<std::size_t>(row))*width+
                   static_cast<std::size_t>(column)] = {c.r,c.i};
      }
}

std::complex<double> RealHarmonics::coefficient(int degree, int row, int order) const
{
  if (degree < 0 || degree > maximum_ || row < 0 || row > 2*degree || std::abs(order) > degree)
    throw std::out_of_range("real harmonic coefficient index out of range");
  const std::size_t width = 2*static_cast<std::size_t>(maximum_)+1;
  return transform_[(static_cast<std::size_t>(degree)*width+static_cast<std::size_t>(row))*width+
                    static_cast<std::size_t>(order+degree)];
}

void RealHarmonics::evaluate(const std::array<double,3>& direction, std::span<double> out) const
{
  const std::size_t count = (static_cast<std::size_t>(maximum_)+1)*(static_cast<std::size_t>(maximum_)+1);
  if (out.size() != count) throw std::invalid_argument("the output of the real harmonics has the wrong size");
  const double x = direction[0], y = direction[1], z = direction[2];
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    throw std::invalid_argument("the direction of the real harmonics must be finite");
  const double rho = std::hypot(x,y);
  const double theta = rho > 0.0 || z != 0.0 ? std::atan2(rho,z) : 0.0;
  const double phi = rho > 0.0 ? std::atan2(y,x) : 0.0;
  const std::size_t width = 2*static_cast<std::size_t>(maximum_)+1;
  std::vector<std::complex<double>> complex_values(width);
  for (int l = 0; l <= maximum_; ++l) {
    for (int m = -l; m <= l; ++m)
      complex_values[static_cast<std::size_t>(m+l)] = basis::integral_harmonic(l,m,theta,phi).value;
    for (int row = 0; row <= 2*l; ++row) {
      const std::complex<double>* t =
          transform_.data()+(static_cast<std::size_t>(l)*width+static_cast<std::size_t>(row))*width;
      double sum = 0.0;
      for (int column = 0; column <= 2*l; ++column)
        sum += t[column].real()*complex_values[static_cast<std::size_t>(column)].real()-
               t[column].imag()*complex_values[static_cast<std::size_t>(column)].imag();
      out[static_cast<std::size_t>(l*l+row)] = sum;
    }
  }
}

RealGauntTable::RealGauntTable(int maximum_left, int maximum_right) : left_(maximum_left), right_(maximum_right)
{
  if (maximum_left < 0 || maximum_right < 0 || maximum_left+maximum_right > maximum_angular_sum)
    throw std::invalid_argument("the angular momenta of a real Gaunt table must be nonnegative with a sum of 19 at most");
  const RealHarmonics harmonics(maximum_left+maximum_right);
  // Nonzero coefficients of the complex harmonics in every real harmonic.
  struct Term { int order; std::complex<double> coefficient; };
  std::vector<std::vector<Term>> terms(static_cast<std::size_t>((left_+right_+1)*(left_+right_+1)));
  for (int l = 0; l <= left_+right_; ++l)
    for (int row = 0; row <= 2*l; ++row)
      for (int m = -l; m <= l; ++m) {
        const std::complex<double> c = harmonics.coefficient(l,row,m);
        if (c != std::complex<double>(0.0,0.0)) terms[static_cast<std::size_t>(l*l+row)].push_back({m,c});
      }

  const int depth = std::min(left_,right_)+1;
  offsets_.assign(static_cast<std::size_t>((left_+1)*(right_+1)*depth)+1,0);
  std::size_t total = 0;
  for (int l1 = 0; l1 <= left_; ++l1)
    for (int l2 = 0; l2 <= right_; ++l2)
      for (int k = 0; k < depth; ++k) {
        offsets_[static_cast<std::size_t>((l1*(right_+1)+l2)*depth+k)] = total;
        const int L = std::abs(l1-l2)+2*k;
        if (L <= l1+l2) total += static_cast<std::size_t>((2*l1+1)*(2*l2+1)*(2*L+1));
      }
  offsets_.back() = total;
  values_.assign(total,0.0);

  for (int l1 = 0; l1 <= left_; ++l1)
    for (int l2 = 0; l2 <= right_; ++l2)
      for (int L = std::abs(l1-l2); L <= l1+l2; L += 2) {
        double* block = values_.data()+index(l1,l2,L);
        for (int r1 = 0; r1 <= 2*l1; ++r1)
          for (int r2 = 0; r2 <= 2*l2; ++r2)
            for (int R = 0; R <= 2*L; ++R) {
              // int Y_{l1 m1} Y_{l2 m2} Y_{L M} dOmega = (-1)^m1 gaunt((l1,-m1); (l2,m2), (L,M)),
              // nonzero for m1 + m2 + M = 0 only.
              std::complex<double> sum(0.0,0.0);
              for (const Term& a : terms[static_cast<std::size_t>(l1*l1+r1)])
                for (const Term& b : terms[static_cast<std::size_t>(l2*l2+r2)])
                  for (const Term& c : terms[static_cast<std::size_t>(L*L+R)]) {
                    if (a.order+b.order+c.order != 0) continue;
                    const double g = basis::gaunt({l1,-a.order},{l2,b.order},{L,c.order});
                    sum += a.coefficient*b.coefficient*c.coefficient*(odd(a.order) ? -g : g);
                  }
              block[(r1*(2*l2+1)+r2)*(2*L+1)+R] = sum.real();
            }
      }
}

std::size_t RealGauntTable::index(int l1, int l2, int L) const
{
  if (l1 < 0 || l1 > left_ || l2 < 0 || l2 > right_ || L < std::abs(l1-l2) || L > l1+l2 || odd(l1+l2+L))
    throw std::out_of_range("real Gaunt block outside the stored triangle");
  const int depth = std::min(left_,right_)+1;
  return offsets_[static_cast<std::size_t>((l1*(right_+1)+l2)*depth+(L-std::abs(l1-l2))/2)];
}

std::span<const double> RealGauntTable::block(int l1, int l2, int L) const
{
  return {values_.data()+index(l1,l2,L),static_cast<std::size_t>((2*l1+1)*(2*l2+1)*(2*L+1))};
}

double RealGauntTable::value(int l1, int r1, int l2, int r2, int L, int R) const
{
  const std::span<const double> values = block(l1,l2,L);
  if (r1 < 0 || r1 > 2*l1 || r2 < 0 || r2 > 2*l2 || R < 0 || R > 2*L)
    throw std::out_of_range("real Gaunt component out of range");
  return values[static_cast<std::size_t>((r1*(2*l2+1)+r2)*(2*L+1)+R)];
}

TwoCenterIntegrator::TwoCenterIntegrator(ReciprocalGrid grid, const Kernel& kernel,
    int maximum_left, int maximum_right)
  : grid_(std::move(grid)), weights_(grid_.size()),
    harmonics_(maximum_left < 0 || maximum_right < 0 || maximum_left+maximum_right > maximum_angular_sum
               ? throw std::invalid_argument("the angular momenta of a two-centre integrator must be nonnegative "
                                             "with a sum of 19 at most")
               : maximum_left+maximum_right),
    gaunt_(maximum_left,maximum_right)
{
  for (std::size_t i = 0; i < grid_.size(); ++i)
    weights_[i] = grid_.weights()[i]*kernel_weight(kernel,grid_.nodes()[i]);
  const int maximum = maximum_left+maximum_right;
  if (maximum < maximum_angular_sum) {
    gradient_harmonics_.emplace(maximum+1);
    gradient_gaunt_.emplace(1,maximum);
  }
}

TwoCenterIntegrator::Pair TwoCenterIntegrator::pair(const std::array<double,3>& displacement) const
{
  Pair result;
  result.owner_ = this;
  result.distance_ = std::hypot(displacement[0],displacement[1],displacement[2]);
  if (!std::isfinite(result.distance_))
    throw std::invalid_argument("the displacement of a two-centre integral must be finite");
  const int maximum = harmonics_.maximum_degree();
  const std::size_t nodes = grid_.size();
  result.bessel_.resize(static_cast<std::size_t>(maximum+1)*nodes);
  for (std::size_t i = 0; i < nodes; ++i) {
    const basis::SphericalBesselSample sample(grid_.nodes()[i]*result.distance_,maximum);
    const std::span<const double> values = sample.values();
    for (int L = 0; L <= maximum; ++L)
      result.bessel_[static_cast<std::size_t>(L)*nodes+i] = weights_[i]*values[static_cast<std::size_t>(L)];
  }
  result.harmonics_.resize(static_cast<std::size_t>((maximum+1)*(maximum+1)));
  harmonics_.evaluate(displacement,result.harmonics_);
  return result;
}

double TwoCenterIntegrator::Pair::radial(const ReciprocalFunction& p, const ReciprocalFunction& q, int order) const
{
  if (!owner_) throw std::logic_error("an empty two-centre pair cannot be evaluated");
  const std::size_t nodes = owner_->grid_.size();
  if (p.values.size() != nodes || q.values.size() != nodes)
    throw std::invalid_argument("a reciprocal function does not match the grid of the integrator");
  if (order < 0 || order > owner_->harmonics_.maximum_degree())
    throw std::out_of_range("the order of a radial two-centre integral is out of range");
  const double* table = bessel_.data()+static_cast<std::size_t>(order)*nodes;
  double sum = 0.0;
  for (std::size_t i = 0; i < nodes; ++i) sum += table[i]*p.values[i]*q.values[i];
  return sum;
}

void TwoCenterIntegrator::Pair::block(const ReciprocalFunction& p, const ReciprocalFunction& q,
    std::span<double> out) const
{
  if (!owner_) throw std::logic_error("an empty two-centre pair cannot be evaluated");
  const int lp = p.angular, lq = q.angular;
  if (lp < 0 || lp > owner_->gaunt_.maximum_left() || lq < 0 || lq > owner_->gaunt_.maximum_right())
    throw std::out_of_range("the angular momentum of a function exceeds the bound of the integrator");
  const int rows = 2*lp+1, columns = 2*lq+1;
  if (out.size() != static_cast<std::size_t>(rows*columns))
    throw std::invalid_argument("the output of a two-centre block has the wrong size");
  std::fill(out.begin(),out.end(),0.0);
  for (int L = std::abs(lp-lq); L <= lp+lq; L += 2) {
    // i^(lp-lq-L) is real because lp-lq-L is even.
    const double factor = (odd((lp-lq-L)/2) ? -8.0 : 8.0)*radial(p,q,L);
    const std::span<const double> theta = owner_->gaunt_.block(lp,lq,L);
    const double* y = harmonics_.data()+static_cast<std::size_t>(L*L);
    const int count = 2*L+1;
    for (int rp = 0; rp < rows; ++rp)
      for (int rq = 0; rq < columns; ++rq) {
        const double* t = theta.data()+static_cast<std::size_t>((rp*columns+rq)*count);
        double sum = 0.0;
        for (int R = 0; R < count; ++R) sum += t[R]*y[R];
        out[static_cast<std::size_t>(rp*columns+rq)] += factor*sum;
      }
  }
}

TwoCenterIntegrator::GradientPair TwoCenterIntegrator::gradient_pair(const std::array<double,3>& displacement) const
{
  if (!gradient_harmonics_ || !gradient_gaunt_)
    throw std::invalid_argument("the derivative of a two-centre integral needs angular momenta with a sum of 18 at most");
  GradientPair result;
  result.owner_ = this;
  result.distance_ = std::hypot(displacement[0],displacement[1],displacement[2]);
  if (!std::isfinite(result.distance_))
    throw std::invalid_argument("the displacement of a two-centre integral must be finite");
  const int maximum = harmonics_.maximum_degree();
  const std::size_t nodes = grid_.size();
  result.bessel_.resize(static_cast<std::size_t>(maximum+2)*nodes);
  for (std::size_t i = 0; i < nodes; ++i) {
    const basis::SphericalBesselSample sample(grid_.nodes()[i]*result.distance_,maximum+1);
    const std::span<const double> values = sample.values();
    const double weight = weights_[i]*grid_.nodes()[i];
    for (int L = 0; L <= maximum+1; ++L)
      result.bessel_[static_cast<std::size_t>(L)*nodes+i] = weight*values[static_cast<std::size_t>(L)];
  }

  std::vector<double> y(static_cast<std::size_t>((maximum+2)*(maximum+2)));
  gradient_harmonics_->evaluate(displacement,y);
  // d^_axis = sqrt(4 pi/3) Y_{1,axis}: the rows of degree 1 are x, y, z.
  const double scale = std::sqrt(4.0*std::numbers::pi/3.0);
  const std::size_t count = static_cast<std::size_t>((maximum+1)*(maximum+1));
  result.raise_.assign(3*count,0.0);
  result.lower_.assign(3*count,0.0);
  for (int L = 0; L <= maximum; ++L)
    for (int neighbour : {L-1,L+1}) {
      if (neighbour < 0) continue;
      const std::span<const double> theta = gradient_gaunt_->block(1,L,neighbour);
      const double* source = y.data()+static_cast<std::size_t>(neighbour*neighbour);
      std::vector<double>& target = neighbour > L ? result.raise_ : result.lower_;
      const int width = 2*neighbour+1;
      for (int axis = 0; axis < 3; ++axis)
        for (int row = 0; row <= 2*L; ++row) {
          const double* t = theta.data()+static_cast<std::size_t>((axis*(2*L+1)+row)*width);
          double sum = 0.0;
          for (int R = 0; R < width; ++R) sum += t[R]*source[R];
          target[static_cast<std::size_t>(axis)*count+static_cast<std::size_t>(L*L+row)] = scale*sum;
        }
    }
  return result;
}

double TwoCenterIntegrator::GradientPair::radial(const ReciprocalFunction& p, const ReciprocalFunction& q,
    int order) const
{
  if (!owner_) throw std::logic_error("an empty two-centre pair cannot be evaluated");
  const std::size_t nodes = owner_->grid_.size();
  if (p.values.size() != nodes || q.values.size() != nodes)
    throw std::invalid_argument("a reciprocal function does not match the grid of the integrator");
  if (order < 0 || order > owner_->harmonics_.maximum_degree()+1)
    throw std::out_of_range("the order of a radial two-centre integral is out of range");
  const double* table = bessel_.data()+static_cast<std::size_t>(order)*nodes;
  double sum = 0.0;
  for (std::size_t i = 0; i < nodes; ++i) sum += table[i]*p.values[i]*q.values[i];
  return sum;
}

void TwoCenterIntegrator::GradientPair::block(const ReciprocalFunction& p, const ReciprocalFunction& q,
    std::span<double> out) const
{
  if (!owner_) throw std::logic_error("an empty two-centre pair cannot be evaluated");
  const int lp = p.angular, lq = q.angular;
  if (lp < 0 || lp > owner_->gaunt_.maximum_left() || lq < 0 || lq > owner_->gaunt_.maximum_right())
    throw std::out_of_range("the angular momentum of a function exceeds the bound of the integrator");
  const int rows = 2*lp+1, columns = 2*lq+1;
  if (out.size() != static_cast<std::size_t>(3*rows*columns))
    throw std::invalid_argument("the output of the derivative of a two-centre block has the wrong size");
  std::fill(out.begin(),out.end(),0.0);
  const int maximum = owner_->harmonics_.maximum_degree();
  const std::size_t total = static_cast<std::size_t>((maximum+1)*(maximum+1));
  double y[2*maximum_angular_sum+1];
  for (int L = std::abs(lp-lq); L <= lp+lq; L += 2) {
    // i^(lp-lq-L) is real because lp-lq-L is even.
    const double sign = odd((lp-lq-L)/2) ? -8.0 : 8.0;
    const double up = sign*radial(p,q,L+1), down = L > 0 ? sign*radial(p,q,L-1) : 0.0;
    const std::span<const double> theta = owner_->gaunt_.block(lp,lq,L);
    const int count = 2*L+1;
    for (int axis = 0; axis < 3; ++axis) {
      const double* raise = raise_.data()+static_cast<std::size_t>(axis)*total+static_cast<std::size_t>(L*L);
      const double* lower = lower_.data()+static_cast<std::size_t>(axis)*total+static_cast<std::size_t>(L*L);
      for (int R = 0; R < count; ++R) y[R] = down*lower[R]-up*raise[R];
      double* target = out.data()+static_cast<std::size_t>(axis*rows*columns);
      for (int rp = 0; rp < rows; ++rp)
        for (int rq = 0; rq < columns; ++rq) {
          const double* t = theta.data()+static_cast<std::size_t>((rp*columns+rq)*count);
          double sum = 0.0;
          for (int R = 0; R < count; ++R) sum += t[R]*y[R];
          target[static_cast<std::size_t>(rp*columns+rq)] += sum;
        }
    }
  }
}

std::vector<double> two_center_matrix(const TwoCenterIntegrator::Pair& pair,
    std::span<const ReciprocalFunction> left, std::span<const ReciprocalFunction> right)
{
  std::size_t rows = 0, columns = 0;
  for (const ReciprocalFunction& f : left) rows += static_cast<std::size_t>(2*f.angular+1);
  for (const ReciprocalFunction& f : right) columns += static_cast<std::size_t>(2*f.angular+1);
  std::vector<double> matrix(rows*columns);
  std::vector<double> block;
  std::size_t row = 0;
  for (const ReciprocalFunction& p : left) {
    const std::size_t block_rows = static_cast<std::size_t>(2*p.angular+1);
    std::size_t column = 0;
    for (const ReciprocalFunction& q : right) {
      const std::size_t block_columns = static_cast<std::size_t>(2*q.angular+1);
      block.resize(block_rows*block_columns);
      pair.block(p,q,block);
      for (std::size_t i = 0; i < block_rows; ++i)
        for (std::size_t j = 0; j < block_columns; ++j)
          matrix[(row+i)*columns+column+j] = block[i*block_columns+j];
      column += block_columns;
    }
    row += block_rows;
  }
  return matrix;
}

std::vector<double> two_center_gradient_matrix(const TwoCenterIntegrator::GradientPair& pair,
    std::span<const ReciprocalFunction> left, std::span<const ReciprocalFunction> right)
{
  std::size_t rows = 0, columns = 0;
  for (const ReciprocalFunction& f : left) rows += static_cast<std::size_t>(2*f.angular+1);
  for (const ReciprocalFunction& f : right) columns += static_cast<std::size_t>(2*f.angular+1);
  std::vector<double> matrix(3*rows*columns);
  std::vector<double> block;
  std::size_t row = 0;
  for (const ReciprocalFunction& p : left) {
    const std::size_t block_rows = static_cast<std::size_t>(2*p.angular+1);
    std::size_t column = 0;
    for (const ReciprocalFunction& q : right) {
      const std::size_t block_columns = static_cast<std::size_t>(2*q.angular+1);
      block.resize(3*block_rows*block_columns);
      pair.block(p,q,block);
      for (std::size_t axis = 0; axis < 3; ++axis)
        for (std::size_t i = 0; i < block_rows; ++i)
          for (std::size_t j = 0; j < block_columns; ++j)
            matrix[(axis*rows+row+i)*columns+column+j] = block[(axis*block_rows+i)*block_columns+j];
      column += block_columns;
    }
    row += block_rows;
  }
  return matrix;
}

} // namespace openmx::exx

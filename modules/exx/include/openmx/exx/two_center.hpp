#pragma once

#include <openmx/exx/kernel.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

// Two-centre integrals between atom-centred functions
//   P(r) = f_P(r) Y_lm(r^),  Y_lm a real spherical harmonic,
// by the Fourier-space formulas of the implementation guide:
//   (11)  f~_P(kappa) = int_0^inf r^2 f_P(r) j_l(kappa r) dr
//   (12)  (P|v|Q)_d = 8 sum_{LM} i^{l_P-l_Q-L} Theta^{LM}_{PQ} Y_LM(d^)
//                       int_0^inf kappa^2 f~_P f~_Q v~(kappa) j_L(kappa d) dkappa
//   (13)  the radial part of the Coulomb potential of P.
// d points from the centre of P to the centre of Q.  Lengths are in bohr.
//
// The derivative of (12) with respect to d uses
//   grad [ j_L(kappa d) Y_LM(d^) ]
//     = kappa ( j_{L-1}(kappa d) [d^ Y_LM]_{L-1} - j_{L+1}(kappa d) [d^ Y_LM]_{L+1} ),
// where [d^ Y_LM]_{L'} is the part of degree L' of the product of the unit
// vector with the harmonic,
//   d^_i Y_LM = sqrt(4 pi/3) sum_{L'M'} Theta(1 i, LM, L'M') Y_L'M',  L' = L-1, L+1.
// No term is divided by d.  The quadrature in kappa is the one of the value,
// so the result is the derivative of the discretized integral.
//
// Real spherical harmonics use the order of the OpenMX orbitals: s; p = (x,
// y, z); d = (z2, x2-y2, xy, xz, yz); for l >= 3: m = 0, cos(phi), sin(phi),
// cos(2 phi), sin(2 phi), ...  A component is addressed by its row 0..2l in
// that order.
//
// The discretization is the one of the overlap integrals of the main program:
// a uniform radial grid and a uniform reciprocal grid, both with the trapezoid
// rule.  Nothing here keeps global state; objects can be used from several
// threads as long as each thread writes to its own output.
namespace openmx::exx {

// Uniform radial grid r_i = first + i*step, i = 0..intervals.  A function on
// the grid is the vector of its values at the nodes and vanishes outside.
struct RadialGrid {
  double first = 0.0;
  double step = 0.0;
  int intervals = 0;

  std::size_t size() const noexcept { return static_cast<std::size_t>(intervals)+1; }
  double radius(std::size_t point) const noexcept { return first+static_cast<double>(point)*step; }
  double last() const noexcept { return radius(static_cast<std::size_t>(intervals)); }
  // Trapezoid weight of a node, without the factor r^2: step, and step/2 at both ends.
  double weight(std::size_t point) const noexcept
  {
    return point == 0 || point == static_cast<std::size_t>(intervals) ? 0.5*step : step;
  }
};

// Throws std::invalid_argument unless first >= 0, step > 0 and intervals >= 3,
// all finite.
void check(const RadialGrid& grid);

// The radial part of an atom-centred function and its angular momentum.
struct RadialFunction {
  int angular = 0;
  std::vector<double> values; // at the nodes of a RadialGrid
};

// int r^2 a(r) b(r) dr with the trapezoid rule of the grid.
double radial_integral(const RadialGrid& grid, std::span<const double> a, std::span<const double> b);

// int r^(power+2) a(r) dr with the trapezoid rule of the grid.
double radial_moment(const RadialGrid& grid, std::span<const double> a, int power);

// Quadrature in the reciprocal coordinate: nodes kappa_i >= 0 in increasing
// order and positive weights.
class ReciprocalGrid {
 public:
  ReciprocalGrid(std::vector<double> nodes, std::vector<double> weights);
  // kappa_i = minimum + i*(maximum-minimum)/intervals, i = 0..intervals, with
  // the trapezoid rule.  The overlap integrals of the main program use
  // minimum = 1e-6, maximum = sqrt(1DFFT.EnergyCutoff), intervals = 1DFFT.NumGridR.
  static ReciprocalGrid uniform(double maximum, int intervals, double minimum = 0.0);

  std::size_t size() const noexcept { return nodes_.size(); }
  std::span<const double> nodes() const noexcept { return nodes_; }
  std::span<const double> weights() const noexcept { return weights_; }

 private:
  std::vector<double> nodes_, weights_;
};

// The spherical Bessel transform (11) of a radial function at the nodes of a
// reciprocal grid.
struct ReciprocalFunction {
  int angular = 0;
  std::vector<double> values;
};

// Equation (11) for several functions on one radial grid.  The sum over the
// radial nodes is the trapezoid rule; angular momenta 0..29 are accepted.
// The cost is (number of radial nodes) x (number of reciprocal nodes) Bessel
// evaluations, shared by all functions.
std::vector<ReciprocalFunction> bessel_transform(const RadialGrid& grid,
    std::span<const RadialFunction> functions, const ReciprocalGrid& reciprocal);
ReciprocalFunction bessel_transform(const RadialGrid& grid, const RadialFunction& function,
    const ReciprocalGrid& reciprocal);

// Equation (13): the radial part w(r) of the Coulomb potential of f(r) Y_lm,
//   w(r) = 4 pi/(2l+1) [ r^(-l-1) int_0^r f r'^(l+2) dr' + r^l int_r^inf f r'^(1-l) dr' ],
// at the nodes of the grid.  The integrals over the grid use a rule that is
// exact for cubic polynomials on every interval (fourth order).  Between the
// origin and the first node f is continued as f(first) (r/first)^l.
RadialFunction coulomb_potential(const RadialGrid& grid, const RadialFunction& function);

// Real spherical harmonics of all degrees 0..maximum_degree (0..29) at a
// direction, in the OpenMX order.
class RealHarmonics {
 public:
  explicit RealHarmonics(int maximum_degree);
  int maximum_degree() const noexcept { return maximum_; }
  // Entry l*l + row holds Y_{l,row}.  The vector need not be normalized; the
  // zero vector is treated as the z axis.  out must hold (maximum_degree+1)^2
  // values.
  void evaluate(const std::array<double,3>& direction, std::span<double> out) const;
  // Coefficient of the complex harmonic Y_l^m (Condon-Shortley phase) in the
  // real harmonic "row" of degree l.
  std::complex<double> coefficient(int degree, int row, int order) const;

 private:
  int maximum_;
  std::vector<std::complex<double>> transform_; // [degree][row][column], width 2*maximum+1
};

// Gaunt coefficients of real spherical harmonics,
//   Theta(l1 r1, l2 r2, L R) = int Y_{l1,r1} Y_{l2,r2} Y_{L,R} dOmega,
// for l1 <= maximum_left, l2 <= maximum_right and every L of the triangle
// |l1-l2| <= L <= l1+l2 with l1+l2+L even (the other L give zero).
// maximum_left + maximum_right must not exceed 19.
class RealGauntTable {
 public:
  RealGauntTable(int maximum_left, int maximum_right);
  int maximum_left() const noexcept { return left_; }
  int maximum_right() const noexcept { return right_; }
  // The block of one (l1, l2, L): entry (r1*(2*l2+1) + r2)*(2*L+1) + R.
  // Throws std::out_of_range outside the stored triangle.
  std::span<const double> block(int l1, int l2, int L) const;
  double value(int l1, int r1, int l2, int r2, int L, int R) const;

 private:
  std::size_t index(int l1, int l2, int L) const;
  int left_, right_;
  std::vector<std::size_t> offsets_;
  std::vector<double> values_;
};

// Equation (12) for one kernel on one reciprocal grid.
class TwoCenterIntegrator {
 public:
  // maximum_left and maximum_right bound the angular momenta of the two
  // functions; their sum must not exceed 19.
  TwoCenterIntegrator(ReciprocalGrid grid, const Kernel& kernel, int maximum_left, int maximum_right);
  // A Pair refers to the integrator that made it, so the integrator stays put.
  TwoCenterIntegrator(const TwoCenterIntegrator&) = delete;
  TwoCenterIntegrator& operator=(const TwoCenterIntegrator&) = delete;

  const ReciprocalGrid& grid() const noexcept { return grid_; }
  int maximum_left() const noexcept { return gaunt_.maximum_left(); }
  int maximum_right() const noexcept { return gaunt_.maximum_right(); }

  // The quantities that depend on the displacement only: j_L(kappa_i d) with
  // the quadrature and kernel weights, and Y_LM(d^).  A Pair must not outlive
  // its integrator.
  class Pair {
   public:
    // int kappa^2 p~ q~ v~ j_L(kappa d) dkappa.
    double radial(const ReciprocalFunction& p, const ReciprocalFunction& q, int order) const;
    // (P|v|Q)_d for all components: out[row_P*(2*l_Q+1) + row_Q].
    void block(const ReciprocalFunction& p, const ReciprocalFunction& q, std::span<double> out) const;
    double distance() const noexcept { return distance_; }

   private:
    friend class TwoCenterIntegrator;
    const TwoCenterIntegrator* owner_ = nullptr;
    double distance_ = 0.0;
    std::vector<double> bessel_;    // [order][node], weights included
    std::vector<double> harmonics_; // [L*L + row]
  };

  Pair pair(const std::array<double,3>& displacement) const;

  // The quantities that the derivative with respect to the displacement
  // needs: j_L(kappa_i d) up to the order maximum_left + maximum_right + 1
  // with the weights of a Pair and one more power of kappa, and the two parts
  // of d^_axis Y_LM at d^.  A GradientPair must not outlive its integrator.
  class GradientPair {
   public:
    // int kappa^3 p~ q~ v~ j_L(kappa d) dkappa.
    double radial(const ReciprocalFunction& p, const ReciprocalFunction& q, int order) const;
    // d(P|v|Q)_d / d d_axis for all components and the axes x, y, z:
    // out[(axis*(2*l_P+1) + row_P)*(2*l_Q+1) + row_Q].
    void block(const ReciprocalFunction& p, const ReciprocalFunction& q, std::span<double> out) const;
    double distance() const noexcept { return distance_; }

   private:
    friend class TwoCenterIntegrator;
    const TwoCenterIntegrator* owner_ = nullptr;
    double distance_ = 0.0;
    std::vector<double> bessel_; // [order][node], weights and kappa included
    std::vector<double> raise_;  // [axis][L*L + row]: the part of degree L+1 of d^_axis Y_{L,row}
    std::vector<double> lower_;  // [axis][L*L + row]: the part of degree L-1
  };

  // Throws std::invalid_argument when maximum_left + maximum_right is 19: the
  // derivative needs harmonics of one more degree.
  GradientPair gradient_pair(const std::array<double,3>& displacement) const;

 private:
  ReciprocalGrid grid_;
  std::vector<double> weights_; // quadrature weight times kappa^2 v~(kappa)
  RealHarmonics harmonics_;
  RealGauntTable gaunt_;
  // For the derivative: harmonics up to maximum_left + maximum_right + 1 and
  // the coefficients Theta(1 i, LM, L'M').  Empty when the sum is 19.
  std::optional<RealHarmonics> gradient_harmonics_;
  std::optional<RealGauntTable> gradient_gaunt_;
};

// The blocks of all pairs of two lists of functions as one matrix.  The rows
// are the components of "left" in their order (2l+1 for every function), the
// columns those of "right": out[row*columns + column].
std::vector<double> two_center_matrix(const TwoCenterIntegrator::Pair& pair,
    std::span<const ReciprocalFunction> left, std::span<const ReciprocalFunction> right);

// The derivative of that matrix with respect to the displacement:
// out[(axis*rows + row)*columns + column].
std::vector<double> two_center_gradient_matrix(const TwoCenterIntegrator::GradientPair& pair,
    std::span<const ReciprocalFunction> left, std::span<const ReciprocalFunction> right);

} // namespace openmx::exx

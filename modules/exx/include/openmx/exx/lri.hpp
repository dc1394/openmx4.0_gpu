#pragma once

#include <openmx/exx/two_center.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// Local resolution of the identity (LRI), sections 5.4 and 5.7 of the
// implementation guide.  The product of a PAO mu on site A and a PAO lambda on
// site C is expanded in the auxiliary basis functions (ABFs) of the two sites,
//   phi_mu phi_lambda ~ sum_{P in A} C^P P + sum_{Q in C} C^Q Q,
// with the coefficients of equation (7),
//   sum_Q G_PQ C^Q = b^P,   G_PQ = (P|g|Q),   b^P = (phi_mu phi_lambda|g|P),
// in the metric g = 1/r.
//
// b^P is evaluated as the guide describes: the Coulomb potential W_P of the
// ABF (equation (13)) times the PAO of the same atom is again an atom-centred
// function, expanded in real spherical harmonics with Gaunt coefficients, and
// its overlap with the other PAO is a two-centre integral (equation (12)).
// Integrals between functions of one atom are radial integrals.
//
// Components are numbered as the orbitals of the main program: the radial
// functions in their order, each with its 2l+1 real components.
namespace openmx::exx {

// Everything of one species that the LRI needs and that does not depend on
// the geometry.
class SpeciesTable {
 public:
  // One term of the expansion of phi w_P in spherical harmonics: the product
  // of the radial functions of orbital shell "orbital" and of the potential of
  // ABF "abf", transformed with the order transform.angular.
  struct Product {
    std::size_t orbital = 0;
    std::size_t abf = 0;
    ReciprocalFunction transform;
  };

  // orbitals: the radial functions of the PAOs in the order of the main
  // program.  abfs: the radial functions of the auxiliary basis, for example
  // make_auxiliary_basis(grid, orbitals, threshold).functions.  All functions
  // are on "grid"; the transforms are made on "reciprocal".  Orbitals may
  // have angular momenta up to 4 and ABFs up to 8.
  SpeciesTable(const RadialGrid& grid, std::vector<RadialFunction> orbitals, std::vector<RadialFunction> abfs,
               const ReciprocalGrid& reciprocal);

  const RadialGrid& grid() const noexcept { return grid_; }
  // The radius beyond which the PAOs and the ABFs vanish.
  double radius() const noexcept { return grid_.last(); }
  std::size_t reciprocal_size() const noexcept { return reciprocal_size_; }

  std::span<const RadialFunction> orbitals() const noexcept { return orbitals_; }
  std::span<const RadialFunction> abfs() const noexcept { return abfs_; }
  // Number of PAOs (u) and of ABFs (n_P) with their components counted.
  std::size_t orbital_count() const noexcept { return orbital_offsets_.back(); }
  std::size_t abf_count() const noexcept { return abf_offsets_.back(); }
  // First component of every radial function; one more entry holds the count.
  std::span<const std::size_t> orbital_offsets() const noexcept { return orbital_offsets_; }
  std::span<const std::size_t> abf_offsets() const noexcept { return abf_offsets_; }
  int maximum_orbital_angular() const noexcept { return maximum_orbital_; }
  int maximum_abf_angular() const noexcept { return maximum_abf_; }

  std::span<const ReciprocalFunction> orbital_transforms() const noexcept { return orbital_transforms_; }
  std::span<const ReciprocalFunction> abf_transforms() const noexcept { return abf_transforms_; }
  // The radial parts w_P of the Coulomb potentials of the ABFs.
  std::span<const RadialFunction> potentials() const noexcept { return potentials_; }
  // The products phi w_P, ordered by orbital shell, ABF and order.
  std::span<const Product> products() const noexcept { return products_; }

  // G_PQ between the ABFs of one atom, n_P x n_P, row-major.
  std::span<const double> onsite_metric() const noexcept { return onsite_metric_; }
  // b^P_{mu lambda} with both PAOs and the ABF on one atom:
  // [(P*u + mu)*u + lambda].
  std::span<const double> onsite_right_hand_side() const noexcept { return onsite_rhs_; }

 private:
  RadialGrid grid_;
  std::size_t reciprocal_size_ = 0;
  std::vector<RadialFunction> orbitals_, abfs_, potentials_;
  std::vector<std::size_t> orbital_offsets_, abf_offsets_;
  int maximum_orbital_ = 0, maximum_abf_ = 0;
  std::vector<ReciprocalFunction> orbital_transforms_, abf_transforms_;
  std::vector<Product> products_;
  std::vector<double> onsite_metric_, onsite_rhs_;
};

// How equation (7) is solved.  The metric is scaled to a unit diagonal first,
// so both numbers are relative to the Coulomb norms of the ABFs.  The
// defaults are the size of the disagreement between the two ways in which
// the metric is evaluated (radial integrals on one atom, equation (12)
// between atoms) on the grids of the main program, about 1e-7: a direction
// below that is not resolved.
struct LriOptions {
  // A Cholesky pivot at or below this value sends the system to the
  // eigen-decomposition.
  double pivot_tolerance = 1e-7;
  // The eigen-decomposition leaves out the eigenvectors of the scaled metric
  // whose eigenvalue is at or below this value.
  double eigenvalue_threshold = 1e-7;
};

// What happened in the solutions of equation (7); merge() adds another report.
struct LriReport {
  long systems = 0;              // linear systems solved
  long eigen_decompositions = 0; // of them, by eigen-decomposition
  long dropped = 0;              // eigenvectors left out in total
  double smallest_pivot = 1.0;   // of the scaled metrics factorized by Cholesky
  void merge(const LriReport& other) noexcept;
};

// Solves G X = B for a symmetric positive definite G (n x n, row-major) and
// several right-hand sides, B[i*m + k] for row i and right-hand side k; X
// replaces B.  G is scaled to a unit diagonal and factorized by Cholesky.  If
// a pivot falls to options.pivot_tolerance, the scaled matrix is diagonalized
// instead and the eigenvectors at or below options.eigenvalue_threshold are
// left out.  Throws std::runtime_error if a diagonal element is not positive.
void solve_metric(std::span<const double> matrix, std::size_t n, std::span<double> right_hand_sides,
                  const LriOptions& options = {}, LriReport* report = nullptr);

// Coefficients of one pair of sites.
struct PairCoefficients {
  // first[(P*u_a + mu)*u_c + lambda]: the coefficient of ABF P of the first
  // site in phi_mu phi_lambda, mu on the first site and lambda on the second.
  std::vector<double> first;
  // second[(Q*u_c + lambda)*u_a + mu]: the coefficient of ABF Q of the second
  // site in the same product.
  std::vector<double> second;
};

class LriSolver {
 public:
  // reciprocal: the grid of the species tables.  maximum_orbital and
  // maximum_abf bound the angular momenta of the PAOs and of the ABFs of all
  // species that will be passed; 2*maximum_orbital + maximum_abf and
  // 2*maximum_abf must not exceed 19.
  LriSolver(const ReciprocalGrid& reciprocal, int maximum_orbital, int maximum_abf, const LriOptions& options = {});

  const LriOptions& options() const noexcept { return options_; }

  // The T block of an atom with itself, [(P*u + mu)*u + lambda]: equation (7)
  // with the ABFs of the atom alone, and half of its solution (section 5.4).
  std::vector<double> onsite(const SpeciesTable& a, LriReport* report = nullptr) const;

  // Equation (7) for mu on a site of species a and lambda on a site of
  // species c; displacement points from the first site to the second one and
  // must not vanish.
  PairCoefficients pair(const SpeciesTable& a, const SpeciesTable& c, const std::array<double,3>& displacement,
                        LriReport* report = nullptr) const;

  // The parts of the system of a pair.  metric: (n_a + n_c)^2, row-major, the
  // ABFs of the first site first.  right_hand_side: [(P*u_a + mu)*u_c +
  // lambda] with P over the ABFs of both sites.
  std::vector<double> metric(const SpeciesTable& a, const SpeciesTable& c,
                             const std::array<double,3>& displacement) const;
  std::vector<double> right_hand_side(const SpeciesTable& a, const SpeciesTable& c,
                                      const std::array<double,3>& displacement) const;

 private:
  void check(const SpeciesTable& table) const;
  LriOptions options_;
  int maximum_orbital_, maximum_abf_;
  TwoCenterIntegrator overlap_, coulomb_;
  RealGauntTable gaunt_;
};

} // namespace openmx::exx

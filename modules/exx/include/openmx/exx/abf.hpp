#pragma once

#include <openmx/exx/two_center.hpp>

#include <cstddef>
#include <span>
#include <vector>

// Auxiliary basis functions (ABFs) of one species, section 6.2 of the
// implementation guide.
//
// The ABFs of a species span the products of the radial functions of its PAOs.
// A product g = phi_i phi_j of functions with angular momenta l_i and l_j has
// components with |l_i - l_j| <= L <= l_i + l_j, l_i + l_j + L even.  For
// every L, the products with such a component are reduced by a principal
// component analysis of
//   O_kk' = int r^2 g_k(r) g_k'(r) dr :
// the eigenvectors whose eigenvalue exceeds a threshold times the largest
// eigenvalue give the radial functions of the ABFs,
//   f_n = sum_k U_kn g_k / sqrt(lambda_n),
// which are orthonormal, int r^2 f_n f_n' dr = delta_nn'.  The sign of f_n is
// the one that makes int r^(L+2) f_n dr positive, so that the ABFs do not
// depend on the machine.
namespace openmx::exx {

// A radial function of the main program on its radial mesh: a PAO of a .pao
// file with the mesh "radii" of the file.
struct MeshFunction {
  int angular = 0;
  std::span<const double> values;
};

// The radial functions at the nodes of a uniform grid, by the interpolation
// that the main program uses for its PAOs.  Beyond the last point of the mesh
// a function vanishes; below the first point it keeps its first value for
// l = 0 and vanishes otherwise.  The mesh must be finite and strictly
// increasing with at least three points, the values finite.
std::vector<RadialFunction> sample_orbitals(const RadialGrid& grid, std::span<const double> mesh,
                                            std::span<const MeshFunction> orbitals);

// One angular momentum of the products.
struct AbfChannel {
  int angular = 0;
  // All eigenvalues of O for this angular momentum, in decreasing order.
  // Their number is the number of products.
  std::vector<double> eigenvalues;
  // The first "kept" eigenvalues exceed threshold * eigenvalues[0].
  std::size_t kept = 0;
};

struct AuxiliaryBasis {
  // Radial functions of the ABFs: L ascending, and within one L by decreasing
  // eigenvalue.  Every function stands for its 2L+1 real components.
  std::vector<RadialFunction> functions;
  // One entry per angular momentum that occurs, ascending.
  std::vector<AbfChannel> channels;
  // Number of ABFs with the 2L+1 components counted.
  std::size_t components() const noexcept;
};

// The ABFs of the radial functions "orbitals" (all on the grid; their order
// and their signs change the result by rounding only).  threshold is
// scf.exx.abf.pca; it must be in [1e-12, 1): below 1e-12 rounding decides
// what a component is.  Angular momenta up to 14 are accepted.
AuxiliaryBasis make_auxiliary_basis(const RadialGrid& grid, std::span<const RadialFunction> orbitals,
                                    double threshold);

} // namespace openmx::exx

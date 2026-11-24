#pragma once

#include <openmx/exx/dense_backend.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/lri.hpp>
#include <openmx/exx/system.hpp>
#include <openmx/exx/two_center.hpp>

#include <cstddef>
#include <span>

// The derivative of E_x with respect to the positions of the atoms at a fixed
// density matrix: section 5.9 of the implementation guide.
//
// E_x depends on the positions through the T and V blocks, and a block of a
// pair of sites depends on the displacement d between the two sites only.
// With the derivatives of E_x with respect to the blocks (equations (16) and
// (17)) the chain rule gives a vector g = dE_x/dd for every pair: equation
// (18) for a T block, the derivative of equation (12) for a V block.  g is
// added to the atom at the end of d and subtracted from the atom at its start,
// so the sum over the atoms vanishes.  A pair of two images of one atom
// contributes nothing: its displacement is a lattice vector.
//
// The neighbour lists count as fixed.  E_x itself jumps when a pair of sites
// crosses a cutoff of the lists; the derivative here is that of the smooth
// function on either side.  The orbital response is not part of it: the
// Pulay term of the main program carries that.
namespace openmx::exx {

class PositionGradient {
 public:
  // The arguments of prepare_system() and its result.  They are kept by
  // reference and must outlive this object.
  PositionGradient(const Structure& structure, std::span<const SpeciesTable> species,
                   const ReciprocalGrid& reciprocal, const SystemOptions& options, const System& system);

  // Adds the part of dE_x/dtau that goes through the stored blocks of one
  // centre atom: gradient[3*b + x] for atom b and the axis x gets its share,
  // three entries for every atom of the unit cell.  weights holds the
  // derivatives of E_x with respect to those blocks, as centre_gradient_blocks()
  // or centre_gradient_gamma() return them.  The parts of all centre atoms
  // add up to dE_x/dtau; different processes may compute them.
  //
  // Every pair of sites is treated in the description that prepare_system()
  // computed, so the system of equation (7) is factorized as it was there.
  // One call factorizes the systems of all C neighbours of the atom: a pair
  // is visited from both of its atoms.
  void add(std::size_t atom, const CentreGradient& weights, std::span<double> gradient) const;

  // centre_gradient_gamma() and add() for one centre atom.  K is factor times
  // the contributions of all centre atoms (-alpha for the exchange matrix),
  // E_x = (1/2) sum D K, and density is N x N and symmetric.
  void add_gamma(std::size_t atom, double factor, std::span<const double> density, std::span<double> gradient,
                 const DenseBackend& backend = DenseBackend()) const;

  // centre_gradient_blocks() and add(): density is an array in the layout,
  // with D_xy(R) = D_yx(-R)^T.
  void add_blocks(const BlockLayout& layout, std::size_t atom, double factor, std::span<const double> density,
                  std::span<double> gradient, const DenseBackend& backend = DenseBackend()) const;

 private:
  const SpeciesTable& table(std::size_t atom) const;

  const Structure* structure_;
  std::span<const SpeciesTable> species_;
  const System* system_;
  LriSolver solver_;
  TwoCenterIntegrator interaction_;
};

} // namespace openmx::exx

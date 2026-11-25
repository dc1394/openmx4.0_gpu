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
//
// Two routes give the same derivative.  add() takes the weights of one centre
// atom and visits the pairs of that atom, so a pair is visited from both of
// its atoms.  add_pairs() takes the weights of all atoms and visits every
// pair once; processes that hold all weights divide the pairs among them
// (section 6.7).
namespace openmx::exx {

// The derivatives of E_x with respect to the blocks of an atom as one array,
// in the order of get_tables(): those of System::t[atom] first, then those of
// System::v[atom][i] for every V neighbour i.  The call adds the derivatives
// of a centre atom (centre_gradient_blocks() or centre_gradient_gamma()) to
// such an array; the arrays of several spin channels add up.
void add_block_weights(const CentreGradient& weights, std::span<double> values);

// The same for a part of the ABFs of the unit cell, which are counted atom
// after atom as for add_exchange_gamma_abfs(): the derivatives of the ABFs
// first <= P < last are computed (centre_gradient_gamma() or
// centre_gradient_blocks() for the part of every atom that the range holds)
// and added to the rows of these ABFs in weights[atom].  weights has an
// entry for every atom of the unit cell; the entries of the atoms that the
// range does not touch are not looked at and may be empty.  The ranges of a
// division of all ABFs fill the arrays of all atoms, each element from one
// range, so processes that divide the ABFs add their arrays up without
// rounding.  K is factor times the contributions of all centre atoms, and
// the density matrix is symmetric.  An empty range adds nothing.
void add_block_weights_gamma_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                                  std::span<const double> density, std::span<const std::span<double>> weights,
                                  const DenseBackend& backend = DenseBackend());
void add_block_weights_blocks_abfs(const System& system, const BlockLayout& layout, std::size_t first,
                                   std::size_t last, double factor, std::span<const double> density,
                                   std::span<const std::span<double>> weights,
                                   const DenseBackend& backend = DenseBackend());

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

  // Adds the part of dE_x/dtau of a share of the pairs of sites.  weights[a]
  // holds the derivatives of E_x with respect to the blocks of atom a, for
  // every atom of the unit cell, in the order of add_block_weights().  Every
  // pair of two different atoms is visited once, in the description that
  // prepare_system() computed: one factorization of the system of equation
  // (7) serves the T blocks of both atoms, and one derivative of the V block
  // serves the two blocks that are stored.  The pairs of the T blocks are
  // numbered in the order of these visits, and so are the pairs of the V
  // blocks; the call treats the pairs whose number leaves the remainder
  // share.part when divided by share.parts.  The parts of a division add up
  // to dE_x/dtau, which add() gives when it is called for every atom, within
  // rounding.
  void add_pairs(std::span<const std::span<const double>> weights, const PrepareShare& share,
                 std::span<double> gradient) const;

 private:
  const SpeciesTable& table(std::size_t atom) const;

  const Structure* structure_;
  std::span<const SpeciesTable> species_;
  const System* system_;
  LriSolver solver_;
  TwoCenterIntegrator interaction_;
};

} // namespace openmx::exx

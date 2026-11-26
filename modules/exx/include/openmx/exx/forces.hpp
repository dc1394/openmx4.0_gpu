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
//
// Both routes go through the pairs of sites of the preparation.  In a system
// whose V neighbours fold_v_neighbours() folded, the weights have the layout
// of the folded blocks, and the pair of a V block of the preparation takes
// the weights of the folded block that holds it (System::folding): E_x
// depends on the blocks of the members of a class through their sum.
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
// the density matrix is symmetric.  An empty range adds nothing.  With a
// device in the backend, the device computes the parts first
// (device_gradients_gamma() or device_gradients_blocks()), and the backend
// computes the parts that the device leaves.
void add_block_weights_gamma_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                                  std::span<const double> density, std::span<const std::span<double>> weights,
                                  const DenseBackend& backend = DenseBackend());
void add_block_weights_blocks_abfs(const System& system, const BlockLayout& layout, std::size_t first,
                                   std::size_t last, double factor, std::span<const double> density,
                                   std::span<const std::span<double>> weights,
                                   const DenseBackend& backend = DenseBackend());

// The half form of these weights (section 6.7 of the implementation guide):
// half_gradient_gamma() or half_gradient_blocks() for every part, the form
// of E_x of the half build.  The V blocks of a pair of sites carry the
// derivative of the pair once, on the block that the half build goes
// through, and the T blocks of an atom also get the parts through W of the
// parts of other atoms.  The derivative through PositionGradient::add_pairs()
// is that of the calls above within rounding.  A range adds to the arrays
// of its atoms and of the atoms of their V neighbours, so weights needs an
// array for every atom of the unit cell unless the range is empty; the
// ranges of a division add to the same elements, and processes that divide
// the ABFs add their arrays up with rounding.  The device computes the parts
// first, as above, and the backend those that the device leaves.  For every
// V neighbour of the half build a part takes three products of the size of
// W instead of two, and the half build goes through about half of them.
void add_block_weights_gamma_half_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                                       std::span<const double> density, std::span<const std::span<double>> weights,
                                       const DenseBackend& backend = DenseBackend());
void add_block_weights_blocks_half_abfs(const System& system, const BlockLayout& layout, std::size_t first,
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
  //
  // The pairs of the share are divided among the pieces of the backend
  // (DenseBackend::pieces()), pair k to piece k mod pieces, and the loop of
  // the backend may run the pieces at the same time.  Every pair keeps its
  // vector g, and the vectors are added to the gradient in the order of the
  // pairs afterwards: the result is the same for every number of pieces,
  // bit for bit.  Only the loop of the backend is used.
  void add_pairs(std::span<const std::span<const double>> weights, const PrepareShare& share,
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

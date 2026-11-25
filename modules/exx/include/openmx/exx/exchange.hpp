#pragma once

#include <openmx/exx/dense_backend.hpp>
#include <openmx/exx/system.hpp>

#include <array>
#include <compare>
#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <vector>

// The exchange matrix from the T and V blocks of a structure and a density
// matrix: sections 5.5 and 6.4 of the implementation guide, with equations
// (4), (15), (16) and (17).  All functions treat one spin channel; the
// density matrix has occupations between 0 and 1.
//
// Two forms are provided.  The site form works with blocks M_xy(R) between
// the orbitals of atom x in the cell at the origin and those of atom y in the
// cell R, and serves any k mesh.  The Gamma form folds all cells onto the
// unit cell and works with N x N arrays, N the number of orbitals of the unit
// cell; the orbitals are ordered by atom.  All arrays are row-major.
namespace openmx::exx {

// ---- site form -------------------------------------------------------------

// The block M_xy(R): rows on atom "row" in the cell at the origin, columns on
// atom "column" in the cell "cell".
struct BlockKey {
  int row = 0;
  int column = 0;
  std::array<int,3> cell{};
  friend auto operator<=>(const BlockKey&, const BlockKey&) = default;
};

// Writes the block D_xy(R) of the density matrix, u_x x u_y values, row-major,
// into a block that holds zeros.  It is called for every block that the
// build reads, some of them more than once.  The build itself assumes no
// symmetry; K_xy(R) is the transpose of K_yx(-R) when the density matrix has
// that property.
using DensityBlocks = std::function<void(const BlockKey& key, std::span<double> block)>;

// Blocks of a matrix M(R), kept in the order of their keys.
class BlockMatrix {
 public:
  using Blocks = std::map<BlockKey,std::vector<double>>;

  // orbital_count[a] is the number of orbitals of atom a.
  explicit BlockMatrix(std::vector<std::size_t> orbital_count);

  // The block of a key, made of zeros if it does not exist yet.  Throws
  // std::out_of_range for an atom that does not exist.
  std::span<double> at(const BlockKey& key);
  // The block of a key; empty if it does not exist.
  std::span<const double> find(const BlockKey& key) const noexcept;

  const Blocks& blocks() const noexcept { return blocks_; }
  std::span<const std::size_t> orbital_count() const noexcept { return orbital_count_; }
  void scale(double factor) noexcept;

 private:
  std::vector<std::size_t> orbital_count_;
  Blocks blocks_;
};

// Adds factor times the contribution of one centre atom to the blocks of K:
// the four terms of section 5.5 for all V neighbours of the atom.  The
// contributions of all atoms with factor = -alpha give the exchange matrix
// of equation (3) in the local RI.
void add_exchange_blocks(const System& system, std::size_t atom, double factor, const DensityBlocks& density,
                         BlockMatrix& k, const DenseBackend& backend = DenseBackend());

// K(R) of one spin channel: all atoms, factor -alpha.
BlockMatrix exchange_blocks(const System& system, const DensityBlocks& density, double alpha,
                            const DenseBackend& backend = DenseBackend());

// Equation (4) for one spin channel: (1/2) sum over the blocks of K of
// sum_{mu nu} D_{mu nu}(R) K_{mu nu}(R).
double exchange_energy(const DensityBlocks& density, const BlockMatrix& k);

// Equation (15) for one spin channel: the sum of D (K - K') over the blocks of
// both matrices; a block that one of them lacks counts as zero.
double exchange_change(const DensityBlocks& density, const BlockMatrix& k, const BlockMatrix& k_previous);

// ---- site form in one array --------------------------------------------------

// The blocks of the site form in a fixed order.  The build of a centre atom A
// reads the blocks D_xy(R) and writes the blocks K_xy(R) whose rows are the
// sites of loc(A) and whose columns are the sites of S(A) (section 6.4).  A
// layout holds the keys of these blocks for all centre atoms, each key once
// and in increasing order, and gives every block its place in one array: the
// block of keys[i] has orbital_count[row] x orbital_count[column] elements,
// row-major, from offset[i] on.  The layout depends on the neighbour lists
// only, so every process that prepares a structure gets the same one, and the
// arrays of the processes can be added element by element.  K_xy(R) is zero
// outside the layout, and the layout holds the key of K_yx(-R) for every key
// of K_xy(R).
struct BlockLayout {
  std::vector<BlockKey> keys;
  std::vector<std::size_t> offset;   // keys.size()+1 entries

  // The number of elements of an array in this layout.
  std::size_t elements() const noexcept { return offset.empty() ? 0 : offset.back(); }
  // The position of a key, or keys.size() if the layout does not hold it.
  std::size_t find(const BlockKey& key) const noexcept;
};

BlockLayout exchange_layout(const System& system);

// add_exchange_blocks() for arrays in a layout of the system: density holds
// the blocks of D, and factor times the contribution of the centre atom is
// added to k.  Both arrays have layout.elements() elements.
void add_exchange_blocks(const System& system, const BlockLayout& layout, std::size_t atom, double factor,
                         std::span<const double> density, std::span<double> k,
                         const DenseBackend& backend = DenseBackend());

// The same for the centre atoms first to last-1, one after the other.  With
// a device in the backend that accepts the work, the device computes all of
// them (section 11 of the implementation guide); otherwise, and after a
// failure of the device, the backend does, atom by atom.  Returns true when
// the device computed them.
bool add_exchange_blocks_range(const System& system, const BlockLayout& layout, std::size_t first, std::size_t last,
                               double factor, std::span<const double> density, std::span<double> k,
                               const DenseBackend& backend = DenseBackend());

// The same for a part of the work that need not end at an atom.  The
// contribution of a centre atom is a sum over its ABFs P: the products with
// the T blocks, the products with the V blocks and the four terms of every V
// neighbour are separate for every P.  The ABFs of all atoms are counted in
// the order of the atoms, atom after atom, from 0 to the sum of abf_count
// minus 1, and the call adds the terms of the ABFs first <= P < last.  Calls
// whose ranges divide the ABFs of the structure add up to the exchange
// matrix: several processes can share one centre atom (section 6.7 of the
// implementation guide).  A range that holds whole atoms only gives the bits
// of add_exchange_blocks_range() for these atoms.  Ranges that cut an atom
// add the sums over its ABFs in another order, so the result differs by
// rounding.  Every call that touches an atom gathers the rows of D that the
// atom needs.
bool add_exchange_blocks_abfs(const System& system, const BlockLayout& layout, std::size_t first, std::size_t last,
                              double factor, std::span<const double> density, std::span<double> k,
                              const DenseBackend& backend = DenseBackend());

// The half build (section 6.4 of the implementation guide).  K is symmetric,
// K_xy(R) = K_yx(-R)^T, when the density matrix is, and the contribution of
// a centre a through its V neighbour (b, R) is the transpose of that of the
// centre b through its V neighbour (a, -R).  The half build takes one pair of
// the two: R > 0 in the order of the components, or R = 0 and b > a; the
// centre itself, b = a at R = 0, enters with half its V block.  Every centre
// then goes through about half its V neighbours, and S(A) holds the sites of
// these only.  Calls whose ranges divide the ABFs add up to X, and K = X +
// X^T (mirror_blocks()), within rounding of add_exchange_blocks_abfs().  The
// density matrix must have D_xy(R) = D_yx(-R)^T, and the system the V
// neighbours and V blocks of prepare_system(): (a, -R) is a V neighbour of b
// with the transpose of the V block of (b, R) of a, and the V block of the
// centre itself is symmetric.  The arguments and the device are those of
// add_exchange_blocks_abfs().
bool add_exchange_blocks_half_abfs(const System& system, const BlockLayout& layout, std::size_t first,
                                   std::size_t last, double factor, std::span<const double> density,
                                   std::span<double> k, const DenseBackend& backend = DenseBackend());

// M_xy(R) + M_yx(-R)^T for every block of the layout, written over the array:
// both elements of a pair that are equal in a matrix with M_xy(R) =
// M_yx(-R)^T get their sum, and the diagonal of M_xx(0) is doubled.
void mirror_blocks(const System& system, const BlockLayout& layout, std::span<double> values);

// For every atom: an estimate of the multiplications that one of its ABFs
// takes in a build with the atom as the centre, in the site form (sites) or
// in the Gamma form, for the half build with half.  It counts the two
// products with the T blocks of the centre and, for every V neighbour of the
// build, the product with its V block and the four terms.  Processes that
// divide a build weigh the ABFs with it.
std::vector<double> abf_costs(const System& system, bool sites, bool half = false);

// The largest difference between an element of M_xy(R) and the element of
// M_yx(-R) that equals it in a matrix with M_xy(R) = M_yx(-R)^T.
double block_asymmetry(const System& system, const BlockLayout& layout, std::span<const double> values);

// Replaces each such pair of elements by its mean.
void symmetrize_blocks(const System& system, const BlockLayout& layout, std::span<double> values);

// ---- Gamma form -------------------------------------------------------------

// The number of orbitals of the unit cell, N.
std::size_t orbital_total(const System& system) noexcept;

// Adds factor times the contribution of one centre atom to k.  density and k
// are N x N.  Lattice images of an atom fold onto the same orbitals.  The
// folding is done before the matrix products: the V blocks of the images of
// an atom are added up, and so are the columns of the T blocks that belong
// to one orbital.  A cell smaller than the cutoffs therefore takes the
// products of a V neighbour once for every atom among the V neighbours of
// the centre, not once for every image.  The build assumes no symmetry of
// the density matrix; the sum over all atoms is symmetric when the density
// matrix is.
void add_exchange_gamma(const System& system, std::size_t atom, double factor, std::span<const double> density,
                        std::span<double> k, const DenseBackend& backend = DenseBackend());

// The same for the centre atoms first to last-1, as add_exchange_blocks_range().
bool add_exchange_gamma_range(const System& system, std::size_t first, std::size_t last, double factor,
                              std::span<const double> density, std::span<double> k,
                              const DenseBackend& backend = DenseBackend());

// The same for the ABFs first <= P < last in the count over all atoms, as
// add_exchange_blocks_abfs().
bool add_exchange_gamma_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                             std::span<const double> density, std::span<double> k,
                             const DenseBackend& backend = DenseBackend());

// The half build for N x N arrays.  The images of an atom b among the V
// neighbours of the centre a are one group, and the group of b for the
// centre a is the transpose of the group of a for the centre b.  The half
// build takes the groups of the atoms b > a and half the group of the centre
// itself, so that a cell smaller than the cutoffs saves products as well.
// The density matrix must be symmetric, the system must have the symmetry
// of add_exchange_blocks_half_abfs(), and K = X + X^T (mirror_gamma()).
bool add_exchange_gamma_half_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                                  std::span<const double> density, std::span<double> k,
                                  const DenseBackend& backend = DenseBackend());

// M + M^T of an N x N array, written over it.
void mirror_gamma(const System& system, std::span<double> values);

// K of one spin channel: all atoms, factor -alpha.  It equals the sum over R
// of the blocks K(R) that the site form gives for D_xy(R) = D_xy.
std::vector<double> exchange_gamma(const System& system, std::span<const double> density, double alpha,
                                   const DenseBackend& backend = DenseBackend());

// Equation (4) for one spin channel: (1/2) sum_ij D_ij K_ij.
double exchange_energy_gamma(std::span<const double> density, std::span<const double> k);

// Equation (15) for one spin channel: sum_ij D_ij (K_ij - K'_ij).
double exchange_change_gamma(std::span<const double> density, std::span<const double> k,
                             std::span<const double> k_previous);

// Derivatives of E_x = (1/2) sum_ij D_ij K_ij at a fixed symmetric density
// matrix, equations (16) and (17).  t and v have the layout of System::t and
// System::v, and every stored element counts as an independent variable: the
// two V blocks of a pair of sites, which are transposes of each other, get a
// derivative each.
struct ExchangeGradient {
  std::vector<std::vector<double>> t;
  std::vector<std::vector<std::vector<double>>> v;
};
ExchangeGradient exchange_gradient_gamma(const System& system, std::span<const double> density, double alpha,
                                         const DenseBackend& backend = DenseBackend());

// The same derivatives for the blocks of one centre atom, from the arrays of
// its own contribution to K: the site form of equations (16) and (17).  K is
// factor times the sum of the contributions of all centre atoms (-alpha for
// the exchange matrix), E_x = (1/2) sum D K, and the density matrix is
// symmetric, D_xy(R) = D_yx(-R)^T.
//   t  the derivative of E_x with respect to the T blocks of the atom, in the
//      layout of System::t[atom].  The T blocks also enter the contributions
//      of the centres that have the atom as a V neighbour; that part equals
//      the part of the atom's own contribution and is included.  It relies on
//      V blocks as prepare_system() stores them, V_ab(R) = V_ba(-R)^T.
//   v  v[i] is the derivative with respect to the V block of the i-th V
//      neighbour, in the layout of System::v[atom][i].  Every stored block is
//      an independent variable, as in ExchangeGradient.
// A change of all stored blocks changes E_x by the sum over the atoms of
// t o dT and of v[i] o dV[i].  One call takes two matrix products of the
// size of those of the build and, for every V neighbour, two products of the
// size of W and four small ones for every ABF of the centre.
struct CentreGradient {
  std::vector<double> t;
  std::vector<std::vector<double>> v;
};

// density is an array in the layout, as for add_exchange_blocks().
CentreGradient centre_gradient_blocks(const System& system, const BlockLayout& layout, std::size_t atom, double factor,
                                      std::span<const double> density, const DenseBackend& backend = DenseBackend());

// density is N x N.  The images of an atom share the folded arrays, so the
// stored blocks of all images of an atom get the same derivative.
CentreGradient centre_gradient_gamma(const System& system, std::size_t atom, double factor,
                                     std::span<const double> density, const DenseBackend& backend = DenseBackend());

// The same derivatives for a part of the ABFs of the centre atom.  They are
// separate for every ABF P of the centre: the row of P in t (u_a rows of the
// T blocks) and in every v[i] needs the T blocks and the V blocks of that P
// only.  These calls compute the ABFs first <= P < last, counted within the
// atom: t and v[i] hold the rows of these ABFs in the arrays of the calls
// above, (last - first) u_a ell_A and (last - first) n_Q elements, in the
// same order.  With first = 0 and last = abf_count[atom] they are the arrays
// of the calls above, bit for bit.  The rows of a division of the ABFs make
// up the arrays of the whole atom within rounding, since the products then
// run over other numbers of rows; several processes can share one centre
// atom (section 6.7 of the implementation guide).  An empty range gives
// empty arrays, one for t and one for every V neighbour.
CentreGradient centre_gradient_blocks(const System& system, const BlockLayout& layout, std::size_t atom,
                                      std::size_t first, std::size_t last, double factor,
                                      std::span<const double> density, const DenseBackend& backend = DenseBackend());
CentreGradient centre_gradient_gamma(const System& system, std::size_t atom, std::size_t first, std::size_t last,
                                     double factor, std::span<const double> density,
                                     const DenseBackend& backend = DenseBackend());

// The same derivatives on the device of the backend (section 11 of the
// implementation guide), for the parts of the centre atoms that the ABFs
// first <= P < last hold.  The ABFs are counted over all atoms, as for
// add_exchange_blocks_abfs(), and a part is the ABFs from <= P < to of one
// atom, counted within the atom.  One call of the device computes the parts
// in the order of the atoms, with D and the T blocks of all atoms on the
// device for the call.  visit(atom, from, to, gradient) receives the
// derivatives of a part, the arrays of centre_gradient_blocks() or
// centre_gradient_gamma() for that part, as soon as they are back on the
// host.  Returns the number of parts visited: all of them, or fewer when the
// backend has no device, the device does not take the work or a step on the
// device fails.  The caller computes the parts after them on the host.  The
// device adds the products in another order than the host, so the arrays
// differ from those of the host by rounding.
using GradientVisit = std::function<void(std::size_t atom, std::size_t from, std::size_t to, CentreGradient&& gradient)>;
std::size_t device_gradients_blocks(const System& system, const BlockLayout& layout, std::size_t first,
                                    std::size_t last, double factor, std::span<const double> density,
                                    const DenseBackend& backend, const GradientVisit& visit);
std::size_t device_gradients_gamma(const System& system, std::size_t first, std::size_t last, double factor,
                                   std::span<const double> density, const DenseBackend& backend,
                                   const GradientVisit& visit);

} // namespace openmx::exx

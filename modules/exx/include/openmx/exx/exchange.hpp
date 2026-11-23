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
// to one orbital.  A cell smaller than the cutoffs therefore takes five
// products for every atom among the V neighbours of the centre, not five
// for every image.  The build assumes no symmetry of the density matrix;
// the sum over all atoms is symmetric when the density matrix is.
void add_exchange_gamma(const System& system, std::size_t atom, double factor, std::span<const double> density,
                        std::span<double> k, const DenseBackend& backend = DenseBackend());

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

} // namespace openmx::exx

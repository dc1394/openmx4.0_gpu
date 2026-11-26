#pragma once

#include <openmx/exx/dense_backend.hpp>
#include <openmx/exx/kernel.hpp>
#include <openmx/exx/lri.hpp>
#include <openmx/exx/two_center.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// The quantities that depend on the structure, section 6.2 of the
// implementation guide: sites, the two neighbour lists, and the T and V
// blocks.  Nothing here uses MPI; one call prepares the whole unit cell, or
// the share of one process in it (PrepareShare).
namespace openmx::exx {

// An atom of the unit cell together with the lattice vector of its cell,
// R = cell[0] a1 + cell[1] a2 + cell[2] a3.  Its position is tau_atom + R.
struct Site {
  int atom = 0;
  std::array<int,3> cell{};
  friend bool operator==(const Site&, const Site&) = default;
};

// The unit cell and its atoms.  Lengths are in bohr, positions Cartesian.
struct Structure {
  std::array<std::array<double,3>,3> lattice{}; // lattice[i] is the vector a_(i+1)
  std::vector<std::array<double,3>> positions;
  std::vector<int> species;                     // index of the species of every atom
};

// tau_site.atom + R - tau_atom: from the atom in the cell at the origin to the
// site.  The site (b, R) seen from a and the site (a, -R) seen from b give
// vectors that differ in sign only, to the last bit.
std::array<double,3> displacement(const Structure& structure, int atom, const Site& site);

// For every atom a of the unit cell: the sites (b, R) whose distance from a
// does not exceed cutoffs[species_a*species_count + species_b].  The atom
// itself comes first; the other sites follow ordered by atom, then by the
// three components of the cell.  The order does not depend on rounding.  The
// lattice vectors are enumerated here and are not limited by any table of
// the main program.  cutoffs must be symmetric.
std::vector<std::vector<Site>> neighbour_lists(const Structure& structure, std::span<const double> cutoffs,
                                               std::size_t species_count);

// Position of a site in a list made by neighbour_lists(); the size of the list
// if it is absent.
std::size_t find_site(std::span<const Site> list, const Site& site);

// A pair of sites has two descriptions: the site (b, R) seen from atom a, and
// the site (a, -R) seen from atom b.  reversed_site() returns the second one
// for the first.
Site reversed_site(int atom, const Site& site) noexcept;

// prepare_system() computes the blocks of a pair from one of its two
// descriptions and derives the blocks of the other atom from them.  True for
// the description that is computed: the smaller atom first, and for two
// images of one atom the cell that is larger than its negative.  Exactly one
// of the two descriptions of a pair of different sites gives true.
bool pair_computed_from(int atom, const Site& site) noexcept;

struct SystemOptions {
  // The interaction v of the exchange: short_range_kernel(omega) for HSE06
  // (the default, with omega = 0.11/bohr), truncated_coulomb_kernel(R_c) for
  // PBE0.
  Kernel interaction = short_range_kernel(0.11);
  // scf.exx.v.cutoff: V blocks are kept up to the larger ABF radius of the two
  // atoms plus this length (bohr).
  double v_cutoff = 8.0;
  LriOptions lri;
};

// The V neighbours of a system that fold_v_neighbours() folded.  prepared[a]
// is the list of V neighbours of atom a that prepare_system() made, and
// index[a][j] is the V neighbour of the folded list whose block took the
// block of prepared[a][j].  The derivative with respect to the positions
// goes through the pairs of sites of the preparation.  period is that of the
// folding, in cells along a_1, a_2 and a_3.  All are empty, and period is 0,
// for a system that is not folded.  columns tells whether the builds of the
// site form fold their columns by the period as well (fold_columns()), and
// antiperiodic gives the directions along which the density matrix of the
// mesh changes its sign over the period.
struct VFolding {
  std::array<int,3> period{};
  std::vector<std::vector<Site>> prepared;
  std::vector<std::vector<std::size_t>> index;
  bool columns = false;
  std::array<bool,3> antiperiodic{};
};

// Neighbour lists and blocks of one structure.
struct System {
  std::vector<std::size_t> orbital_count; // u_a for every atom
  std::vector<std::size_t> abf_count;     // n_P,a for every atom
  // C neighbours: the sites whose PAOs overlap those of the atom (distance up
  // to the sum of the two PAO radii).  V neighbours: the sites whose V block
  // is kept.  Both lists start with the atom itself.
  std::vector<std::vector<Site>> c_neighbours, v_neighbours;
  // The T blocks of an atom are stored side by side, as section 6.4 uses
  // them: c_offset[a][i] is the first column of the i-th C neighbour of atom a,
  // and c_offset[a].back() is the number of columns ell_a, the number of
  // orbitals on all C neighbours.
  std::vector<std::vector<std::size_t>> c_offset;
  // t[a][(P*u_a + mu)*ell_a + c_offset[a][i] + lambda] = C^(1)P_{mu lambda}:
  // P and mu on atom a, lambda on its i-th C neighbour.  The columns of the
  // atom itself (i = 0) hold half of the solution of equation (7) for the
  // atom alone.
  std::vector<std::vector<double>> t;
  // v[a][i][P*n_b + Q] = (P|v|Q): P on atom a, Q on its i-th V neighbour (atom b).
  std::vector<std::vector<std::vector<double>>> v;
  LriReport report;
  VFolding folding;
};

// The T block of an atom and one of its C neighbours as a separate array,
// block[(P*u_a + mu)*u_c + lambda].
std::vector<double> t_block(const System& system, std::size_t atom, std::size_t neighbour);

// The share of one process in a preparation that several processes divide
// (section 6.7).  prepare_system() goes through the pairs of sites of the T
// blocks in a fixed order, and through those of the V blocks; a share holds
// the pairs whose number in that order leaves the remainder "part" when
// divided by "parts".  The blocks of an atom with itself belong to part 0.
struct PrepareShare {
  std::size_t part = 0, parts = 1;
};

// Section 6.2 for one structure.  species[s] is the table of species s; all
// tables must have been made on "reciprocal".  The ABF radius of a species is
// the radius of its table.  Each pair of sites is solved once and its solution
// goes into the T blocks of both atoms; a V block is computed once and its
// transpose goes to the other atom.
//
// With a share of several parts the call computes the blocks of its share
// only.  The neighbour lists and the sizes of all blocks are those of the
// whole preparation, the other blocks hold zeros, and the report counts the
// systems of the share.  Every element of every block is computed by one
// part, with the operations of the whole preparation: the sum of the blocks
// of all parts, element by element, is the system of the whole preparation
// (get_tables() and set_tables()).
//
// The pairs of the share are divided among the pieces of the backend
// (DenseBackend::pieces()), pair k to piece k mod pieces, and the loop of the
// backend may run the pieces at the same time.  A pair writes its own
// blocks, and the reports of the pieces are merged afterwards, so the system
// is the same for every number of pieces, bit for bit.  Only the loop of the
// backend is used.
System prepare_system(const Structure& structure, std::span<const SpeciesTable> species,
                      const ReciprocalGrid& reciprocal, const SystemOptions& options, const PrepareShare& share = {},
                      const DenseBackend& backend = DenseBackend());

// Section 6.4 of the implementation guide: the V neighbours of a prepared
// system, folded by the period of a k mesh of n_1 x n_2 x n_3 points.
//
// The density matrix of a regular mesh, D_xy(R) = Re sum_k w_k
// e^{2 pi i k.R} D_xy(k), repeats with the period n_i cells along a_i:
// D_xy(R + nT) = s_T D_xy(R), nT = (n_1 T_1, n_2 T_2, n_3 T_3).  The sign
// s_T is e^{2 pi i k.nT}, the same for every point of the mesh: -1 to the
// power T_i along a direction where the mesh is shifted by half a point
// (REGULAR with an even n_i), and 1 otherwise.  The contribution of a
// centre a through its V neighbour (b, R + nT) reads D at the columns of
// loc(b, R + nT) and writes K there, and K enters K(k) at the points of the
// mesh with the phase e^{2 pi i k.(R + nT)}.  The sign of D and the sign of
// the phase cancel, so at the points of the mesh the contribution equals
// that through (b, R) with the same V block.  The contribution is linear in
// the V block.
//
// The call therefore replaces the V neighbours of an atom whose cells agree
// modulo the period, a class, by one site of the class with the sum of
// their V blocks: a build goes through one V neighbour per class.  K(k) at
// the points of the mesh, the energy of a density matrix of the mesh and
// the derivatives with respect to the blocks and the positions keep their
// values within rounding.  K(R) changes, and so does K(k) away from the
// points of the mesh: an output that needs them builds K with the system of
// the preparation.
//
// The site of a class of the centre a is the member closest to a, the
// larger cell among members at the same distance, and the classes of two
// atoms are chosen together: when a < b gets (b, R) for a class, b gets
// (a, -R) for the class of the reversed sites, with the transpose of the
// block.  The classes of images of the atom itself pair up in the same way.
// The class of the atom itself, R = 0 modulo the period, folds onto the atom
// with V(0) plus V(R) + V(R)^T for its members R > 0, a symmetric block.  A
// class of images that is its own mirror image (2R = 0 modulo the period)
// keeps two sites, (a, R) with the sum of the blocks of its members R > 0
// and (a, -R) with the transpose.  The folded system thus has the symmetry
// that the half build requires.  Its lists keep the order of
// neighbour_lists(): the atom itself first, then by atom and cell.  A class
// with one member keeps its block, bit for bit, and so does a class of the
// two images (a, R) and (a, -R): a system without other classes stays as it
// was.
//
// The structure must be that of the preparation, and the blocks of a
// divided preparation must have been added up.  The tables of the atoms
// (get_tables()) hold the folded blocks afterwards.  Throws for a system
// that is folded already and for a period below 1.
void fold_v_neighbours(System& system, const Structure& structure, const std::array<int,3>& period);

// Section 6.4: the columns of the builds of the site form of a system whose
// V neighbours are folded, folded by the same period.  The columns of
// D[loc(A), S] of two sites of S(A) whose cells agree modulo the period, a
// class, differ by the sign s_T only: D_xy(R + nT) = s_T D_xy(R).  A build
// then takes one site of every class into the products with the T blocks of
// the centre, G = T D[loc(A), .] and the last product, the first site of
// the class in the order of S(A), and reads the columns of the others as s_T
// times those of that site.  The last product adds the contributions of the
// class, each times its sign, to the blocks of K of that site: the phase of
// a block at R + nT at a point of the mesh is s_T times that at R, so K(k)
// at the points of the mesh keeps its value, and so do E_x and the
// derivatives, within rounding.  Terms (i) and (ii) still add to the blocks
// of their own sites.  A class of one site gives the bits of the build
// without the folding.
//
// antiperiodic[i] holds along a_i where the points of the mesh lie half a
// point away from the Gamma point, n_i k_i = 1/2 modulo 1 (REGULAR with an
// even number of points): s_T is -1 to the power of the sum of T_i over
// these directions.  A wrong direction gives a wrong K.  Throws for a
// system whose V neighbours are not folded and for one whose columns are
// folded already.
void fold_columns(System& system, const std::array<bool,3>& antiperiodic);

// The T blocks and the V blocks of an atom as one array: System::t[atom]
// first, then System::v[atom][i] for every V neighbour i.  The processes of a
// divided preparation add up these arrays.
std::size_t table_elements(const System& system, std::size_t atom);
// Copies the blocks of the atom into values, table_elements() numbers.
void get_tables(const System& system, std::size_t atom, std::span<double> values);
// Replaces the blocks of the atom by values.
void set_tables(System& system, std::size_t atom, std::span<const double> values);

} // namespace openmx::exx
